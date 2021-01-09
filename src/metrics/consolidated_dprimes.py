import src.data.rasters
from src.data import LDA as cLDA, dPCA as cdPCA
import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import numpy as np
from joblib import Memory, dump, load

import src.data.rasters
from src.data import dPCA as cdPCA
from src.data.cache import set_name
from src.data.load import load, get_site_ids
from src.metrics import dprime as cDP
from src.metrics.reliability import signal_reliability
from src.utils.tools import shuffle_along_axis as shuffle

config = ConfigParser()
config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))

memory = Memory(str(pl.Path(config['paths']['analysis_cache']) / 'prm_dprimes_v2'))

# private functions of snipets of code common to all dprime calculations
def _load_raster(site, probes, meta):
    recs = load(site, rasterfs=meta['raster_fs'], recache=rec_recache)
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
    raster = src.data.rasters.raster_from_sig(sig, probes, channels=goodcells, transitions=meta['transitions'],
                                              smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                              zscore=meta['zscore'], part='probe')

    return raster, goodcells


def  _significance(array, mont_array, multiple_comparisons_axis, alpha=0.01, tails='both'):

    mont_num = mont_array.shape[0] # number of montecarlo repetitions

    if tails is 'both':
        #ToDo is this right??
        top_pval = np.sum((mont_array >= array), axis=0) / mont_num
        bottom_pval = np.sum((mont_array <= array), axis=0) / mont_num
        pvalues = np.where(array >= np.mean(mont_array, axis=0), top_pval, bottom_pval)
    elif tails == 'greater':
        pvalues = np.sum((mont_array >= array), axis=0) / mont_num
    elif tails == 'lesser':
        pvalues = np.sum((mont_array >= array), axis=0) / mont_num
    else:
        raise ValueError("tails must be 'greater' 'lesser' or 'both'")

    significance = pvalues <= alpha

    n_comparisons = np.prod(np.asarray(significance.shape)[np.asarray(multiple_comparisons_axis)])
    n_chance_comp = n_comparisons * alpha

    # count the number of significant bins pooling acrooss the multiple conparions axis
    sig_count = np.sum(significance, axis=tuple(multiple_comparisons_axis), keepdims=True)

    # creates a corrected significance by taking the significant bins of a groups of multiple comparisons, if the sum
    # of significant bins is over the chance comparisons threshold
    # ToDo this is where the next step is unclear, what do we need to do when we pass or fail the treshold?
    corrected_signif = np.where(sig_count > n_chance_comp, significance, np.full(significance.shape, False))
    print(np.allclose(corrected_signif, significance))

    # defines the confidence intervals as the top and bottom percentiles summing to alpha
    low = alpha * 100
    high = (1 - alpha) * 100
    confidence_interval = np.percentile(mont_array, [low, high], axis=0)

    return significance, corrected_signif, confidence_interval




# these functionse should operate over site, and a probe, and return a pairwise dprime between contexts, plus the shuffled
# and simulated montecarlos and  good cells

def single_cell_dprimes(site, probes, meta):
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
    raster, goodcells = _load_raster(site, probes, meta)

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, _, _ = cdPCA.format_raster(raster)
    rep, chn, ctx, prb, tme = trialR.shape
    transition_pairs = list(itt.combinations(meta['transitions'], 2))


    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, condition_axis=2,
                                  flip=meta['dprime_absolute'])  # shape Cell x CtxPair x Probe x Time

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle, then calculates the pairwise dprime

    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)
    shuffled_dprime = np.empty([meta['montecarlo'], chn, len(transition_pairs), prb, tme])
    for pair_idx, tp in enumerate(transition_pairs):
        shuf_trialR = np.empty((meta['montecarlo'], rep, chn, 2, prb, tme))

        tran_idx = np.array([meta['transitions'].index(t) for t in tp])
        ctx_shuffle = trialR[:, :, tran_idx, ...].copy()

        for rr in range(meta['montecarlo']):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)

        shuffled_dprime[:, :, pair_idx, :, :] = cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=3,
                                                                     flip=meta['dprime_absolute'] ).squeeze(axis=2)

    return dprime, shuffled_dprime, goodcells


def probewise_dPCA_dprimes(site, probes, meta):
    """
    performs dimensionality reduction with dPCA done independently for each probe. Then uses the first context dependent
    demixed principal component to calculated the dprime between context for all probes in the site. Calculates
    significance using montecarlo shuffling of the context identity.
    :param site: string  identifying the site
    :param probes: list of probe numbers
    :param meta: dict with meta parameters
    :return: dprime (ndarray with shape Ctx_pair x Probe x Time),
             shuffled_dprimes (ndarray with shape Montecarlo x Ctx_pair x Probe x Time),
             goocells (list of strings)
    """
    raster, goodcells = _load_raster(site, probes, meta)

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    rep, unt, ctx, prb, tme = trialR.shape
    transition_pairs = list(itt.combinations(meta['transitions'], 2))

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

            tran_idx = np.array([meta['transitions'].index(t) for t in tp])
            ctx_shuffle = dPCA_projection[:, tran_idx, :].copy()

            for rr in range(meta['montecarlo']):
                shuf_projections[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=1, indie_axis=0, rng=rng)

            shuffled_dprime[:, pair_idx, probe_idx, :] = cDP.pairwise_dprimes(shuf_projections,
                                                                              observation_axis=1,
                                                                              condition_axis=2,
                                                                              flip=meta['dprime_absolute']
                                                                              ).squeeze(axis=1)

    return dprime, shuffled_dprime, goodcells


def probewise_LDA_dprimes(site, probes, meta):
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
    raster, goodcells = _load_raster(site, probes, meta)

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    rep, unt, ctx, prb, tme = trialR.shape
    transition_pairs = list(itt.combinations(meta['transitions'], 2))

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

            tran_idx = np.array([meta['transitions'].index(t) for t in tp])
            ctx_shuffle = LDA_projection[:, tran_idx, :].copy()

            for rr in range(meta['montecarlo']):
                shuf_projections[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=1, indie_axis=0, rng=rng)

            shuffled_dprime[:, pair_idx, probe_idx, :] = cDP.pairwise_dprimes(shuf_projections,
                                                                              observation_axis=1,
                                                                              condition_axis=2,
                                                                              flip=meta['dprime_absolute']
                                                                              ).squeeze(axis=1)

    return dprime, shuffled_dprime, goodcells


def full_dPCA_dprimes(site, probes, meta):
    raster, goodcells = _load_raster(site, probes, meta)

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
    transition_pairs = list(itt.combinations(meta['transitions'], 2))

    shuffled_dprimes = np.empty([meta['montecarlo'], len(transition_pairs), prb, tme])
    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)
    for pair_idx, tp in enumerate(transition_pairs):
        shuf_projections = np.empty([meta['montecarlo'], rep, 2, prb, tme])

        tran_idx = np.array([meta['transitions'].index(t) for t in tp])
        ctx_shuffle = dPCA_projection[:, tran_idx, :, :].copy()

        for rr in range(meta['montecarlo']):
            shuf_projections[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=1, indie_axis=0, rng=rng)

        shuffled_dprimes[:, pair_idx, :, :] = cDP.pairwise_dprimes(shuf_projections, observation_axis=1, condition_axis=2,
                                                                   flip=meta['dprime_absolute']).squeeze(axis=1)

    return dprime, shuffled_dprimes, goodcells

rec_recache = False
site = 'CRD004a'
sites = [site]
# all_probes = [2, 3, 5, 6] #triplets
all_probes = [1, 2, 3, 4] #permutations
probes = all_probes

# meta = {'reliability': 0.1,  # r value
#         'smoothing_window': 0,  # ms
#         'raster_fs': 30,
#         'transitions': ['silence', 'continuous', 'similar', 'sharp'],
#         'montecarlo': 1000,
#         'zscore': True,
#         'dprime_absolute': None,
#         'stim_type': 'triplets'}
#
meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': [0,1,2,3,4],
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None,
        'stim_type': 'permutations'}

# single_cell_dprimes(site, probes, meta)
# probewise_dPCA_dprimes(site, probes, meta)
# probewise_LDA_dprimes(site, probes, meta)

dprime, shuffled_dprime, _ = full_dPCA_dprimes(site, probes, meta)


significance, corrected_signif, confidence_interval = _significance(dprime,shuffled_dprime, [0, 1, 2])

import matplotlib.pyplot as plt
from src.visualization.fancy_plots import _significance_bars, _cint

fig, ax = plt.subplots()
t = np.arange(dprime.shape[-1])
ax.plot(t, dprime[0,0,:], color='black')
ax.fill_between(t, confidence_interval[0,0,0,:], confidence_interval[1,0,0,:], color='black', alpha=0.5)
ax_bottom = ax.get_ylim()[0]
ax.bar(t, significance[0, 0, :], width=1, align='center', color='blue', edgecolor='black', bottom=ax_bottom)
# _ = _cint(t, shuffled_dprime[:, 0, 0, :], confidence=0.95, ax=ax,
#                fillkwargs={'color': 'black', 'alpha': 0.5})



