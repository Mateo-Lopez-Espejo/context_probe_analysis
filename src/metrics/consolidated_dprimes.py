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

full_dPCA_dprimes(site, probes, meta)

