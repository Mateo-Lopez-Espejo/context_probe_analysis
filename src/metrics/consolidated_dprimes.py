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
    :param site:
    :param probe:
    :param meta:
    :return:
    """
    raster, goodcells = _load_raster(site, probes, meta)

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, _, _ = cdPCA.format_raster(raster)
    rep, chn, ctx, prb, tme = trialR.shape
    shuf_shape = (meta['montecarlo'], rep, chn, 2, prb, tme)


    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, condition_axis=2,
                                  flip=meta['dprime_absolute'])  # shape Cell x CtxPair x Probe x Time

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle
    # calculates the pairwise dprime

    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)
    shuffled_dprimes = list()
    for tp in itt.combinations(meta['transitions'], 2):
        shuf_trialR = np.empty(shuf_shape)
        shuf_trialR[:] = np.nan

        tran_idx = np.array([meta['transitions'].index(t) for t in tp])
        ctx_shuffle = trialR[:, :, tran_idx, ...].copy()

        for rr in range(meta['montecarlo']):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)

        shuffled_dprimes.append(cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=3,
                                                     flip=meta['dprime_absolute']))

    # staks the transitions pairs into the same dimension as the original transisions i.e. dim 2
    shuffled_dprimes = np.stack(shuffled_dprimes, axis=2).squeeze(axis=3) # shape Montecarlo x Unit x Ctx_pair x Probe x Time
    return dprime, shuffled_dprimes, goodcells

def dPCA_fourway_analysis(site, probes, meta):
    raster, goodcells = _load_raster(site, probes, meta, stim_type='triplets')

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    trialR, R = trialR.squeeze(axis=3), R.squeeze(axis=2)  # squeezes out probe
    Re, C, S, T = trialR.shape

    # calculates full dPCA. i.e. considering all 4 categories
    dPCA_projection, dPCA_transformation = cdPCA.fit_transform(R, trialR)
    dprime = cDP.pairwise_dprimes(dPCA_projection, observation_axis=0, condition_axis=1,
                                  flip=meta['dprime_absolute'])

    # calculates floor (ctx shuffle) and ceiling (simulated data)
    sim_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))
    shuf_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))

    # ctx_shuffle = trialR.copy()
    shuf_projection = dPCA_projection.copy()

    for rr in range(meta['montecarlo']):
        # ceiling: simulates data, calculates dprimes
        sim_trial = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0),
                                     size=[Re, C, S, T])
        sim_projection = cdPCA.transform(sim_trial, dPCA_transformation)
        sim_dprime[rr, ...] = cDP.pairwise_dprimes(sim_projection, observation_axis=0, condition_axis=1,
                                                   flip=meta['dprime_absolute'])

        # ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)
        # shuf_projection = cdPCA.transform(ctx_shuffle, dPCA_transformation)
        shuf_projection = shuffle(shuf_projection, shuffle_axis=1, indie_axis=0)
        shuf_dprime[rr, ...] = cDP.pairwise_dprimes(shuf_projection, observation_axis=0, condition_axis=1,
                                                    flip=meta['dprime_absolute'])

    return dprime, shuf_dprime, sim_dprime, goodcells

def LDA_fourway_analysis(site, probes, meta):
    raster, goodcells = _load_raster(site, probes, meta, stim_type='triplets')

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, _, _ = cdPCA.format_raster(raster)
    trialR = trialR.squeeze(axis=3)  # squeezes out probe
    Re, C, S, T = trialR.shape

    # calculates full LDA. i.e. considering all 4 categories
    LDA_projection, LDA_transformation = cLDA.fit_transform_over_time(trialR, 1)
    LDA_projection = LDA_projection.squeeze(axis=1)
    dprime = cDP.pairwise_dprimes(LDA_projection, observation_axis=0, condition_axis=1,
                                  flip=meta['dprime_absolute'])

    # calculates floor (ctx shuffle) and ceiling (simulated data)
    sim_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))
    shuf_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))

    ctx_shuffle = trialR.copy()
    # shuf_projection = LDA_projection.copy()

    for rr in range(meta['montecarlo']):
        # ceiling: simulates data, calculates dprimes
        sim_trial = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0),
                                     size=[Re, C, S, T])
        # sim_projection = cLDA.transform_over_time(sim_trial, LDA_transformation).squeeze(axis=1)
        sim_projection, _ = cLDA.fit_transform_over_time(sim_trial)
        sim_projection = sim_projection.squeeze(axis=1)
        sim_dprime[rr, ...] = cDP.pairwise_dprimes(sim_projection, observation_axis=0, condition_axis=1,
                                                   flip=meta['dprime_absolute'])

        ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)
        shuf_projection, _ = cLDA.fit_transform_over_time(ctx_shuffle)
        shuf_projection = shuf_projection.squeeze(axis=1)
        # shuf_projection = shuffle(shuf_projection, shuffle_axis=1, indie_axis=0)
        shuf_dprime[rr, ...] = cDP.pairwise_dprimes(shuf_projection, observation_axis=0, condition_axis=1,
                                                    flip=meta['dprime_absolute'])

    return dprime, shuf_dprime, sim_dprime, goodcells

def probewise_dPCA_dprimes(site, probes, meta):
    raster, goodcells = _load_raster(site, probes, meta, stim_type='permuations')

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    trialR, R = trialR.squeeze(axis=3), R.squeeze(axis=2)  # squeezes out probe

    # calculates full dPCA. i.e. considering all 4 categories
    _, trialZ, _ = cdPCA._cpp_dPCA(R, trialR)
    dPCA_projection = trialZ['ct'][:, 0, ...]
    dprime = cDP.pairwise_dprimes(dPCA_projection, observation_axis=0, condition_axis=1,
                                  flip=meta['dprime_absolute'])

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle
    # calculates the pairwise dprime
    rep, ctx, tme = dPCA_projection.shape
    shuffled_dprimes = list()
    trans_pairs = [f'{x}_{y}' for x, y in itt.combinations(meta['transitions'], 2)]
    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)
    for tp in trans_pairs:
        shuf_projections = np.empty([meta['montecarlo'], rep, 2, tme])
        shuf_projections[:] = np.nan

        tran_idx = np.array([meta['transitions'].index(int(t)) for t in tp.split('_')])
        ctx_shuffle = dPCA_projection[:, tran_idx, :].copy()

        for rr in range(meta['montecarlo']):
            shuf_projections[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=1, indie_axis=0, rng=rng)

        shuffled_dprimes.append(cDP.pairwise_dprimes(shuf_projections, observation_axis=1, condition_axis=2,
                                                     flip=meta['dprime_absolute']))

    shuffled_dprimes = np.stack(shuffled_dprimes, axis=2).squeeze(axis=0)  # shape Montecarlo x ContextPair x Time

    return dprime, shuffled_dprimes, goodcells

def full_dPCA_dprimes(site, probes, meta):
    raster, goodcells = _load_raster(site, probes, meta, stim_type='permuations')

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
    shuffled_dprimes = list()
    trans_pairs = [f'{x}_{y}' for x, y in itt.combinations(meta['transitions'], 2)]
    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)
    for tp in trans_pairs:
        shuf_projections = np.empty([meta['montecarlo'], rep, 2, prb, tme])
        shuf_projections[:] = np.nan

        tran_idx = np.array([meta['transitions'].index(int(t)) for t in tp.split('_')])
        ctx_shuffle = dPCA_projection[:, tran_idx, :, :].copy()

        for rr in range(meta['montecarlo']):
            shuf_projections[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=1, indie_axis=0, rng=rng)

        shuffled_dprimes.append(cDP.pairwise_dprimes(shuf_projections, observation_axis=1, condition_axis=2,
                                                     flip=meta['dprime_absolute']))

    shuffled_dprimes = np.stack(shuffled_dprimes, axis=3).squeeze(axis=0)  # shape Montecarlo x Probe x Ctx_pair x Time
    dprime = np.moveaxis(dprime, 1, 0)  # shape Probe x Ctx_pair x Time

    return dprime, shuffled_dprimes, goodcells

rec_recache = False
site = 'CRD004a'
sites = [site]
all_probes = [2, 3, 5, 6] #triplets
all_probes = [1,2,3,4]
probe = [1]
stim_type = 'triplets'

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None,
        'stim_type': 'triplets'}

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': [0,1,2,3,4],
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None,
        'stim_type': 'permutations'}



single_cell_dprimes(site, probe, meta)

