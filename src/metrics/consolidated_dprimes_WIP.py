from collections import defaultdict
import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import numpy as np
from joblib import Memory

from src.data.rasters import raster_from_sig
from src.data import LDA as cLDA, dPCA as cdPCA
from src.data.load import load_with_parms
from src.metrics import dprime as cDP
from src.metrics.reliability import signal_reliability
from src.metrics.significance import _signif_quantiles
from src.utils.tools import shuffle_along_axis as shuffle

config = ConfigParser()
config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))

memory = Memory(str(pl.Path(config['paths']['analysis_cache']) / 'consolidated_dprimes'))

# private functions of snipets of code common to all dprime calculations
def _load_site_formated_raste(site, contexts, probes, meta, recache_rec=False):
    """
    wrapper of wrappers. Load a recording, selects the subset of data (triplets, or permutations), generates raster using
    selected  probes and transitions
    :param site:
    :param meta:
    :param recache_rec:
    :return:
    """

    recs, _ = load_with_parms(site, rasterfs=meta['raster_fs'], recache=recache_rec)
    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    # pulls the right recording depending on stimulus type and pulls the signal from it.
    if meta['stim_type'] == 'triplets':
        type_key = 'trip0'
    elif meta['stim_type'] == 'permutations':
        type_key = 'perm0'
    else:
        raise ValueError(f"unknown stim type, use 'triplets' or 'permutations'")

    sig = recs[type_key]['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_sequence*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = raster_from_sig(sig, probes=probes, channels=goodcells, contexts=contexts,
                                              smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                              zscore=meta['zscore'], part='probe')

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)

    return trialR, R, goodcells


# these functionse should operate over site, and a probe, and return a pairwise dprime between contexts, plus the shuffled
# and simulated montecarlos and  good cells
@memory.cache
def single_cell_dprimes(site, contexts, probes, meta):
    """
    calculated the dprime between context for all probes and for all cells in a site. Calculates significance using
    montecarlo shuffling of the context identity
    :param site: string  identifying the site
    :param probes: list of probe numbers
    :param meta: dict with meta parameters
    :return: dprime (ndarray with shape Unit x Ctx_pair x Probe x Time),
             shuffled_dprimes (ndarray with shape Montecarlo x Unit x Ctx_pair x Probe x Time),
             goocells (list of strings)
    """
    trialR, R, goodcells = _load_site_formated_raste(site, contexts, probes, meta)

    rep, chn, ctx, prb, tme = trialR.shape
    transition_pairs = list(itt.combinations(contexts, 2))

    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, condition_axis=2,
                                  flip=meta['dprime_absolute'])  # shape Cell x CtxPair x Probe x Time

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle, then calculates the pairwise dprime

    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)

    # does shuffling on a cell by cell basis to deal with memory limits
    shuff_dprime_quantiles = defaultdict(list)

    for cc in range(chn):

        shuf_trialR = np.empty((meta['montecarlo'], rep, ctx, prb, tme))
        ctx_shuffle = trialR[:, cc, :, :, :].copy()

        for rr in range(meta['montecarlo']):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=1, indie_axis=0, rng=rng)

        neur_shuff_dprime = cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=2,
                                               flip=meta['dprime_absolute'])

        neur_shuff_quantils = _signif_quantiles(neur_shuff_dprime)

        for alpha, qntls in neur_shuff_quantils.items():
            shuff_dprime_quantiles[alpha].append(qntls)

    shuff_dprime_quantiles = {alpha: np.stack(qntls, axis=1) for alpha, qntls in shuff_dprime_quantiles.items()}
    return dprime, shuff_dprime_quantiles, goodcells, None


@memory.cache
def probewise_dPCA_dprimes(site, contexts, probes, meta):
    """
    performs dimensionality reduction with dPCA done independently for each probe. Then uses the first context dependent
    demixed principal component to calculated the dprime between context for all probes in the site. Calculates
    significance using montecarlo shuffling of the context identity.
    :param site: string  identifying the site
    :param probes: list of probe numbers
    :param meta: dict with meta parameters
    :return: dprime (ndarray with shape PC x Ctx_pair x Probe x Time),
             shuffled_dprimes (ndarray with shape Montecarlo x PC x Ctx_pair x Probe x Time),
             goocells (list of strings)
    """
    trialR, R, goodcells = _load_site_formated_raste(site, contexts, probes, meta)

    rep, unt, ctx, prb, tme = trialR.shape
    transition_pairs = list(itt.combinations(contexts, 2))

    # iterates over each probe
    dprime = np.empty([len(transition_pairs), prb, tme])
    shuffled_dprime = np.empty([meta['montecarlo'], len(transition_pairs), prb, tme])

    var_capt = list()
    for pp in probes:
        probe_idx = probes.index(pp)
        probe_trialR = trialR[..., probe_idx, :]
        probe_R = R[..., probe_idx, :]

        # calculates dPCA considering all 4 categories
        _, trialZ, dpca = cdPCA._cpp_dPCA(probe_R, probe_trialR)
        var_capt.append(cdPCA.variance_captured(dpca, probe_R))
        dPCA_projection = trialZ['ct'][:, 0, ...]
        dprime[:, probe_idx, :] = cDP.pairwise_dprimes(dPCA_projection, observation_axis=0, condition_axis=1,
                                                       flip=meta['dprime_absolute'])

        # Shuffles the rasters n times and organizes in an array with the same shape as the original raster plus one
        # dimension with size meta['montecarlo'] containing each shuffle
        # calculates the pairwise dprime
        print(f"\nshuffling {meta['montecarlo']} times")
        rng = np.random.default_rng(42)

        shuf_projections = np.empty([meta['montecarlo'], rep, ctx, tme])
        shuf_projections[:] = np.nan

        shuf_trialR = probe_trialR.copy()
        for rr in range(meta['montecarlo']):
            shuf_trialR = shuffle(shuf_trialR, shuffle_axis=2, indie_axis=0, rng=rng)
            shuf_R = np.mean(shuf_trialR, axis=0)

            #saves the first regularizer to speed things up.
            if rr == 0:
                _, shuf_trialZ, shuf_dpca = cdPCA._cpp_dPCA(shuf_R, shuf_trialR)
                regularizer = shuf_dpca.regularizer
            else:
                _, shuf_trialZ, dpca = cdPCA._cpp_dPCA(shuf_R, shuf_trialR, {'regularizer':regularizer})

            shuf_projections[rr, :] = shuf_trialZ['ct'][:, 0, ...]

        shuffled_dprime[:, :, probe_idx, :] = cDP.pairwise_dprimes(shuf_projections,
                                                                   observation_axis=1,
                                                                   condition_axis=2,
                                                                   flip=meta['dprime_absolute'])

    # add dimension for single PC, for compatibility with single cell arrays
    dprime = np.expand_dims(dprime, axis=0)
    shuffled_dprime = np.expand_dims(shuffled_dprime, axis=1)
    shuff_dprime_quantiles = _signif_quantiles(shuffled_dprime)

    return dprime, shuff_dprime_quantiles, goodcells, var_capt


@memory.cache
def probewise_LDA_dprimes(site, contexts, probes, meta):
    """
    performs dimensionality reduction with LDA done independently for each probe. the uses the discriminated projection
    to calculate the dprime between context for all probes in the site. Calculates
    significance using montecarlo shuffling of the context identity.
    :param site: string  identifying the site
    :param probes: list of probe numbers
    :param meta: dict with meta parameters
    :return: dprime (ndarray with shape Ctx_pair x Probe x Time),
             shuffled_dprimes (ndarray with shape Montecarlo x Ctx_pair x Probe x Time),
             goocells (list of strings)
    """
    trialR, R, goodcells = _load_site_formated_raste(site, contexts, probes, meta)

    rep, unt, ctx, prb, tme = trialR.shape
    transition_pairs = list(itt.combinations(contexts, 2))

    # iterates over each probe
    dprime = np.empty([len(transition_pairs), prb, tme])
    shuffled_dprime = np.empty([meta['montecarlo'], len(transition_pairs), prb, tme])

    for pp in probes:
        probe_idx = probes.index(pp)
        probe_trialR = trialR[..., probe_idx, :]

        # calculates LDA considering all 4 categories
        LDA_projection, _ = cLDA.fit_transform_over_time(probe_trialR)
        LDA_projection = LDA_projection.squeeze(axis=1)  # shape Trial x Context x Time
        dprime[:, probe_idx, :] = cDP.pairwise_dprimes(LDA_projection, observation_axis=0, condition_axis=1,
                                                       flip=meta['dprime_absolute'])

        # Shuffles the rasters n times and organizes in an array with the same shape as the original raster plus one
        # dimension with size meta['montecarlo'] containing each shuffle
        # calculates the pairwise dprime
        print(f"\nshuffling {meta['montecarlo']} times")
        rng = np.random.default_rng(42)

        shuf_projections = np.empty([meta['montecarlo'], rep, ctx, tme])
        shuf_projections[:] = np.nan

        shuf_trialR = probe_trialR.copy()

        for rr in range(meta['montecarlo']):
            shuf_trialR = shuffle(shuf_trialR, shuffle_axis=2, indie_axis=0, rng=rng)
            shuf_LDA_projection, _ = cLDA.fit_transform_over_time(shuf_trialR)
            shuf_projections[rr, ...] = shuf_LDA_projection.squeeze(axis=1)  # shape Trial x Context x Time

        shuffled_dprime[:, :, probe_idx, :] = cDP.pairwise_dprimes(shuf_projections,
                                                                   observation_axis=1,
                                                                   condition_axis=2,
                                                                   flip=meta['dprime_absolute'])
    # add dimension for single PC, for compatibility with single cell arrays
    dprime = np.expand_dims(dprime, axis=0)
    shuffled_dprime = np.expand_dims(shuffled_dprime, axis=1)
    shuff_dprime_quantiles = _signif_quantiles(shuffled_dprime)

    return dprime, shuff_dprime_quantiles, goodcells, None


@memory.cache
def full_dPCA_dprimes(site, contexts, probes, meta):
    trialR, R, goodcells = _load_site_formated_raste(site, contexts, probes, meta)

    # calculates full dPCA. i.e. considering all 4 categories
    _, trialZ, dpca = cdPCA._cpp_dPCA(R, trialR)
    dPCA_projection = trialZ['ct'][:, 0, ...]
    dprime = cDP.pairwise_dprimes(dPCA_projection, observation_axis=0, condition_axis=1,
                                  flip=meta['dprime_absolute'])

    # calculates the variance explained. special case for full dpca, not present in other dprime approaches
    var_capt = cdPCA.variance_captured(dpca, R)

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle
    # calculates the pairwise dprime
    rep, ctx, prb, tme = dPCA_projection.shape
    transition_pairs = list(itt.combinations(contexts, 2))

    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)

    shuf_projections = np.empty([meta['montecarlo'], rep, ctx, prb, tme])

    shuf_trialR = trialR.copy()

    for rr in range(meta['montecarlo']):
        shuf_trialR = shuffle(shuf_trialR, shuffle_axis=2, indie_axis=0, rng=rng)
        shuf_R = np.mean(shuf_trialR, axis=0)

        # saves the first regularizer to speed things up.
        if rr == 0:
            _, shuf_trialZ, shuf_dpca = cdPCA._cpp_dPCA(shuf_R, shuf_trialR)
            regularizer = shuf_dpca.regularizer
        else:
            _, shuf_trialZ, dpca = cdPCA._cpp_dPCA(shuf_R, shuf_trialR, {'regularizer': regularizer})

        shuf_projections[rr, ...] = shuf_trialZ['ct'][:, 0, ...]

    shuffled_dprime = cDP.pairwise_dprimes(shuf_projections, observation_axis=1, condition_axis=2,
                                           flip=meta['dprime_absolute'])

    # add dimension for single PC, for compatibility with single cell arrays
    dprime = np.expand_dims(dprime, axis=0)
    shuffled_dprime = np.expand_dims(shuffled_dprime, axis=1)
    shuff_dprime_quantiles = _signif_quantiles(shuffled_dprime)

    return dprime, shuff_dprime_quantiles, goodcells, var_capt

