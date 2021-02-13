import src.data.rasters
from src.data import LDA as cLDA, dPCA as cdPCA
import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import numpy as np
from joblib import Memory

import src.data.rasters
from src.data.load import load
from src.metrics import dprime as cDP
from src.metrics.reliability import signal_reliability
from src.metrics.significance import _significance
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

    recs = load(site, rasterfs=meta['raster_fs'], recache=recache_rec)
    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    # pulls the right recording depending on stimulus type and pulls the signal from it.
    if meta['stim_type'] == 'triplets':
        type_key = 'trip0'
    elif meta['stim_type'] == 'permutations':
        type_key = 'perm0'
    else:
        raise ValueError(f"unknown stim type, use 'triplets' or 'permutations'" )

    sig = recs[type_key]['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = src.data.rasters.raster_from_sig(sig, probes=probes, channels=goodcells, contexts=contexts,
                                              smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                              zscore=meta['zscore'], part='probe')

    return raster, goodcells


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
    raster, goodcells = _load_site_formated_raste(site, contexts, probes, meta)

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, _, _ = cdPCA.format_raster(raster)
    rep, chn, ctx, prb, tme = trialR.shape
    transition_pairs = list(itt.combinations(contexts, 2))


    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, condition_axis=2,
                                  flip=meta['dprime_absolute'])  # shape Cell x CtxPair x Probe x Time

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle, then calculates the pairwise dprime

    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)
    shuffled_dprime = np.empty([meta['montecarlo'], chn, len(transition_pairs), prb, tme])
    for pair_idx, tp in enumerate(transition_pairs):
        shuf_trialR = np.empty((meta['montecarlo'], rep, chn, 2, prb, tme))

        tran_idx = np.array([contexts.index(t) for t in tp])
        ctx_shuffle = trialR[:, :, tran_idx, ...].copy()

        for rr in range(meta['montecarlo']):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)

        shuffled_dprime[:, :, pair_idx, :, :] = cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=3,
                                                                     flip=meta['dprime_absolute'] ).squeeze(axis=2)

    return dprime, shuffled_dprime, goodcells

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
    raster, goodcells = _load_site_formated_raste(site, contexts, probes, meta)

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    rep, unt, ctx, prb, tme = trialR.shape
    transition_pairs = list(itt.combinations(contexts, 2))

    # iterates over each probe
    dprime = np.empty([len(transition_pairs), prb, tme])
    shuffled_dprime = np.empty([meta['montecarlo'], len(transition_pairs), prb, tme])

    for pp in probes:
        probe_idx = probes.index(pp)
        probe_trialR = trialR[..., probe_idx, :]
        probe_R = R[..., probe_idx, :]

        # calculates dPCA considering all 4 categories
        _, trialZ, _ = cdPCA._cpp_dPCA(probe_R, probe_trialR)
        dPCA_projection = trialZ['ct'][:, 0, ...]
        dprime[:, probe_idx, :] = cDP.pairwise_dprimes(dPCA_projection, observation_axis=0, condition_axis=1,
                                                       flip=meta['dprime_absolute'])

        # Shuffles the rasters n times and organizes in an array with the same shape as the original raster plus one
        # dimension with size meta['montecarlo'] containing each shuffle
        # calculates the pairwise dprime
        print(f"\nshuffling {meta['montecarlo']} times")
        rng = np.random.default_rng(42)
        for pair_idx, tp in enumerate(transition_pairs):
            shuf_projections = np.empty([meta['montecarlo'], rep, 2, tme])
            shuf_projections[:] = np.nan

            tran_idx = np.array([contexts.index(t) for t in tp])
            ctx_shuffle = dPCA_projection[:, tran_idx, :].copy()

            for rr in range(meta['montecarlo']):
                shuf_projections[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=1, indie_axis=0, rng=rng)

            shuffled_dprime[:, pair_idx, probe_idx, :] = cDP.pairwise_dprimes(shuf_projections,
                                                                              observation_axis=1,
                                                                              condition_axis=2,
                                                                              flip=meta['dprime_absolute']
                                                                              ).squeeze(axis=1)

    # add dimension for single PC, for compatibility with single cell arrays
    dprime = np.expand_dims(dprime, axis=0)
    shuffled_dprime = np.expand_dims(shuffled_dprime, axis=1)

    return dprime, shuffled_dprime, goodcells

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
    raster, goodcells = _load_site_formated_raste(site, contexts, probes, meta)

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    rep, unt, ctx, prb, tme = trialR.shape
    transition_pairs = list(itt.combinations(contexts, 2))

    # iterates over each probe
    dprime = np.empty([len(transition_pairs), prb, tme])
    shuffled_dprime = np.empty([meta['montecarlo'], len(transition_pairs), prb, tme])

    for pp in probes:
        probe_idx = probes.index(pp)
        probe_trialR = trialR[..., probe_idx, :]
        probe_R = R[..., probe_idx, :]

        # calculates LDA considering all 4 categories
        LDA_projection, _ = cLDA.fit_transform_over_time(probe_trialR)
        LDA_projection = LDA_projection.squeeze(axis=1) # shape Trial x Context x Time
        dprime[:, probe_idx, :] = cDP.pairwise_dprimes(LDA_projection, observation_axis=0, condition_axis=1,
                                                       flip=meta['dprime_absolute'])

        # Shuffles the rasters n times and organizes in an array with the same shape as the original raster plus one
        # dimension with size meta['montecarlo'] containing each shuffle
        # calculates the pairwise dprime
        print(f"\nshuffling {meta['montecarlo']} times")
        rng = np.random.default_rng(42)
        for pair_idx, tp in enumerate(transition_pairs):
            shuf_projections = np.empty([meta['montecarlo'], rep, 2, tme])
            shuf_projections[:] = np.nan

            tran_idx = np.array([contexts.index(t) for t in tp])
            ctx_shuffle = LDA_projection[:, tran_idx, :].copy()

            for rr in range(meta['montecarlo']):
                shuf_projections[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=1, indie_axis=0, rng=rng)

            shuffled_dprime[:, pair_idx, probe_idx, :] = cDP.pairwise_dprimes(shuf_projections,
                                                                              observation_axis=1,
                                                                              condition_axis=2,
                                                                              flip=meta['dprime_absolute']
                                                                              ).squeeze(axis=1)
    # add dimension for single PC, for compatibility with single cell arrays
    dprime = np.expand_dims(dprime, axis=0)
    shuffled_dprime = np.expand_dims(shuffled_dprime, axis=1)

    return dprime, shuffled_dprime, goodcells

@memory.cache
def full_dPCA_dprimes(site, contexts, probes, meta):
    raster, goodcells = _load_site_formated_raste(site, contexts, probes, meta)

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)

    # calculates full dPCA. i.e. considering all 4 categories
    _, trialZ, _ = cdPCA._cpp_dPCA(R, trialR)
    dPCA_projection = trialZ['ct'][:, 0, ...]
    dprime = cDP.pairwise_dprimes(dPCA_projection, observation_axis=0, condition_axis=1,
                                  flip=meta['dprime_absolute'])

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle
    # calculates the pairwise dprime
    rep, ctx, prb, tme = dPCA_projection.shape
    transition_pairs = list(itt.combinations(contexts, 2))

    shuffled_dprime = np.empty([meta['montecarlo'], len(transition_pairs), prb, tme])
    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)
    for pair_idx, tp in enumerate(transition_pairs):
        shuf_projections = np.empty([meta['montecarlo'], rep, 2, prb, tme])

        tran_idx = np.array([contexts.index(t) for t in tp])
        ctx_shuffle = dPCA_projection[:, tran_idx, :, :].copy()

        for rr in range(meta['montecarlo']):
            shuf_projections[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=1, indie_axis=0, rng=rng)

        shuffled_dprime[:, pair_idx, :, :] = cDP.pairwise_dprimes(shuf_projections, observation_axis=1, condition_axis=2,
                                                                   flip=meta['dprime_absolute']).squeeze(axis=1)

    # add dimension for single PC, for compatibility with single cell arrays
    dprime = np.expand_dims(dprime, axis=0)
    shuffled_dprime = np.expand_dims(shuffled_dprime, axis=1)

    return dprime, shuffled_dprime, goodcells


# site = 'CRD004a'
# probes = [1, 2, 3, 4] #permutations
# # probes = [2, 3, 5, 6] #triplets
# contexts = [0, 1, 2, 3, 4]
# # contexts = ['silence', 'continuous', 'similar', 'sharp']
#
#
# meta = {'reliability': 0.1,  # r value
#         'smoothing_window': 0,  # ms
#         'raster_fs': 30,
#         'montecarlo': 1000,
#         'zscore': True,
#         'dprime_absolute': None,
#         'stim_type': 'permutations'}
#
# meta = {'reliability': 0.1,  # r value
#         'smoothing_window': 0,  # ms
#         'raster_fs': 30,
#         'montecarlo': 1000,
#         'zscore': True,
#         'dprime_absolute': None,
#         'stim_type': 'triplets'}
#
# dprime, shuffled_dprime, _ = single_cell_dprimes(site, contexts, probes, meta)
# dprime, shuffled_dprime, _ = probewise_dPCA_dprimes(site, contexts, probes, meta)
# dprime, shuffled_dprime, _ = probewise_LDA_dprimes(site, contexts, probes, meta)
# dprime, shuffled_dprime, _ = full_dPCA_dprimes(site, contexts, probes, meta)
#
# significance, corrected_signif, confidence_interval = _significance(dprime, shuffled_dprime, [1, 2, 3], alpha=0.01)
#
