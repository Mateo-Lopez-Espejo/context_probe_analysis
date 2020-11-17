import collections as col
import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import numpy as np
from joblib import Memory, dump, load

import src.data.rasters
from src.data import dPCA as cdPCA
from src.data.cache import set_name
from src.data.load import load
from src.metrics import dprime as cDP
from src.metrics.reliability import signal_reliability
from src.utils.tools import shuffle_along_axis as shuffle

config = ConfigParser()
config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))

memory = Memory(str(pl.Path(config['paths']['analysis_cache']) / 'prm_dprimes_v2'))


@memory.cache
def single_cell_dprimes(site, probe, meta):
    recs = load(site, rasterfs=meta['raster_fs'], recache=rec_recache)
    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['perm0']
    sig = rec['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = src.data.rasters.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                              smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                              zscore=meta['zscore'], part='probe')

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, _, _ = cdPCA.format_raster(raster)

    rep, chn, ctx, prb, tme = trialR.shape

    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, condition_axis=2,
                                  flip=meta['dprime_absolute'])  # shape CtxPair x Cell x Probe x Time

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle
    # calculates the pairwise dprime
    shuffled_dprimes = list()
    trans_pairs = [f'{x}_{y}' for x, y in itt.combinations(meta['transitions'], 2)]
    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)
    for tp in trans_pairs:
        shuf_trialR = np.empty([meta['montecarlo'], rep, chn, 2, prb, tme])
        shuf_trialR[:] = np.nan

        tran_idx = np.array([meta['transitions'].index(int(t)) for t in tp.split('_')])
        ctx_shuffle = trialR[:, :, tran_idx, :, :].copy()

        for rr in range(meta['montecarlo']):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)

        shuffled_dprimes.append(cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=3,
                                                     flip=meta['dprime_absolute']))

    shuffled_dprimes = np.moveaxis(np.stack(shuffled_dprimes, axis=2).squeeze(axis=0),
                                   3, 1)  # shape Montecarlo x Probe x Ctx_pair x Unit x Time
    dprime = np.moveaxis(dprime, 2, 0)  # shape Probe x Ctx_pair x Unit x Time

    return dprime, shuffled_dprimes, goodcells


@memory.cache
def probewise_dPCA_dprimes(site, probe, meta):
    recs = load(site, rasterfs=meta['raster_fs'], recache=rec_recache)

    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['perm0']
    sig = rec['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = src.data.rasters.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                              smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                              zscore=meta['zscore'])

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

    # #test plots
    # import matplotlib.pyplot as plt
    # import src.visualization.fancy_plots as fplt
    # fig, axes = plt.subplots(2,10)
    # t = np.arange(30)
    # for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
    #
    #     t0_idx = meta['transitions'].index(trans[0])
    #     t1_idx = meta['transitions'].index(trans[1])
    #
    #     axes[0,tt].plot(t, dPCA_projection[:, t0_idx, :].mean(axis=0), color='blue', linewidth=3)
    #     axes[0,tt].plot(t, dPCA_projection[:, t1_idx, :].mean(axis=0), color='orange', linewidth=3)
    #
    # # Raster, dprime, CI
    # bottom, top = axes[0, 0].get_ylim()
    # half = ((top - bottom) / 2) + bottom
    # for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
    #     pair_idx = tt
    #
    #     t0_idx = meta['transitions'].index(trans[0])
    #     t1_idx = meta['transitions'].index(trans[1])
    #
    #     _ = fplt._raster(t, dPCA_projection[:, t0_idx, :], y_offset=0, y_range=(bottom, half), ax=axes[0,tt],
    #                      scatter_kws={'color': 'blue', 'alpha': 0.4, 's': 10})
    #     _ = fplt._raster(t, dPCA_projection[:, t1_idx, :], y_offset=0, y_range=(half, top), ax=axes[0,tt],
    #                      scatter_kws={'color': 'orange', 'alpha': 0.4, 's': 10})
    #
    #     # plots the real dprime and the shuffled dprime
    #     axes[1,tt].plot(t, dprime[pair_idx, :], color='black')
    #     _ = fplt._cint(t, shuffled_dprimes[:, pair_idx, :], confidence=0.95, ax=axes[1,tt],
    #                    fillkwargs={'color': 'black', 'alpha': 0.5})

    return dprime, shuffled_dprimes, goodcells


@memory.cache
def full_dPCA_dprimes(site, probe, meta):
    recs = load(site, rasterfs=meta['raster_fs'], recache=rec_recache)

    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['perm0']
    sig = rec['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = src.data.rasters.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                              smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                              zscore=meta['zscore'])

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

    # # test plots
    # import matplotlib.pyplot as plt
    # import src.visualization.fancy_plots as fplt
    # fig, axes = plt.subplots(2,10)
    # t = np.arange(30)
    # pp = 0
    # for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
    #
    #     t0_idx = meta['transitions'].index(trans[0])
    #     t1_idx = meta['transitions'].index(trans[1])
    #
    #     axes[0,tt].plot(t, dPCA_projection[:, t0_idx, pp, :].mean(axis=0), color='blue', linewidth=3)
    #     axes[0,tt].plot(t, dPCA_projection[:, t1_idx, pp, :].mean(axis=0), color='orange', linewidth=3)
    #
    # # Raster, dprime, CI
    # bottom, top = axes[0, 0].get_ylim()
    # half = ((top - bottom) / 2) + bottom
    # for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
    #     pair_idx = tt
    #
    #     t0_idx = meta['transitions'].index(trans[0])
    #     t1_idx = meta['transitions'].index(trans[1])
    #
    #     _ = fplt._raster(t, dPCA_projection[:, t0_idx, pp,:], y_offset=0, y_range=(bottom, half), ax=axes[0,tt],
    #                      scatter_kws={'color': 'blue', 'alpha': 0.4, 's': 10})
    #     _ = fplt._raster(t, dPCA_projection[:, t1_idx, pp,:], y_offset=0, y_range=(half, top), ax=axes[0,tt],
    #                      scatter_kws={'color': 'orange', 'alpha': 0.4, 's': 10})
    #
    #     # plots the real dprime and the shuffled dprime
    #     axes[1,tt].plot(t, dprime[pp, pair_idx, :], color='black')
    #     _ = fplt._cint(t, shuffled_dprimes[:, pp, pair_idx, :], confidence=0.95, ax=axes[1,tt],
    #                    fillkwargs={'color': 'black', 'alpha': 0.5})

    return dprime, shuffled_dprimes, goodcells


meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': [0, 1, 2, 3, 4],
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None}

dprime_recache = False
rec_recache = False
two_tail_p = True

if dprime_recache: memory.clear()

all_probes = [1, 2, 3, 4]

# sites = set(get_site_ids(316).keys())
badsites = {'AMT031a'}  # empirically deciced
sites = {'AMT028b', 'AMT029a', 'AMT030a', 'AMT031a', 'AMT032a', 'CRD002a', 'CRD003b', 'CRD004a',
         'ley070a', 'ley072b'}
sites = sites.difference(badsites)

analysis_functions = {'SC': single_cell_dprimes, 'pdPCA': probewise_dPCA_dprimes, 'fdPCA': full_dPCA_dprimes}

# initilizede nested dictionary with three layer: 1. Analysis type 2. calculated values 3. cell or site
batch_dprimes = col.defaultdict(lambda: col.defaultdict(dict))
for site, (func_key, func) in itt.product(sites, analysis_functions.items()):

    # for probewise dPCA runs the function probe by probe and stacks the results in an array with a probe dimension
    if func_key == 'pdPCA':

        real_dprimes = list()
        shuffled_dprimes = list()
        pvalues = list()
        for pp, probe in enumerate(all_probes):

            dprime, shuf_dprime, cell_names = func(site, probe, meta)

            real_dprimes.append(dprime)
            shuffled_dprimes.append(shuf_dprime)

            # p value calculations
            if two_tail_p is True:
                top_pval = np.sum((shuf_dprime >= dprime), axis=0) / meta['montecarlo']
                bottom_pval = np.sum((shuf_dprime <= dprime), axis=0) / meta['montecarlo']
                probe_pvalue = np.where(dprime >= np.mean(shuf_dprime, axis=0), top_pval, bottom_pval)
                pvalues.append(probe_pvalue)
            elif two_tail_p is False:
                probe_pvalue = np.sum((shuf_dprime >= dprime), axis=0) / meta['montecarlo']
                pvalues.append(probe_pvalue)

        # appends probe arrays and  organizes into data structure
        batch_dprimes[func_key]['dprime'][site] = np.stack(real_dprimes)
        batch_dprimes[func_key]['shuffled_dprime'][site] = np.stack(shuffled_dprimes, axis=1)
        batch_dprimes[func_key]['shuffled_pvalue'][site] = np.stack(pvalues)

    elif func_key in ['SC', 'fdPCA']:
        real_dprimes, shuffled_dprimes, cell_names = func(site, all_probes,
                                                          meta)  # shape (Montecarlo) x Ctx_pair x Unit x Probe x Time

        # p value calculations
        if two_tail_p is True:
            top_pval = np.sum((shuffled_dprimes >= real_dprimes), axis=0) / meta['montecarlo']
            bottom_pval = np.sum((shuffled_dprimes <= real_dprimes), axis=0) / meta['montecarlo']
            pvalues = np.where(real_dprimes >= np.mean(shuffled_dprimes, axis=0), top_pval, bottom_pval)
        elif two_tail_p is False:
            pvalues = np.sum((shuffled_dprimes >= real_dprimes), axis=0) / meta['montecarlo']

        # appends probe arrays and  organizes into data structure
        batch_dprimes[func_key]['dprime'][site] = real_dprimes
        batch_dprimes[func_key]['shuffled_dprime'][site] = shuffled_dprimes
        batch_dprimes[func_key]['pvalue'][site] = pvalues

        # for single cell analysis further divides the value by cell name
    if func_key == 'SC':
        for source in ['dprime', 'shuffled_dprime', 'pvalue']:
            for (cc, cell) in enumerate(cell_names):
                batch_dprimes[func_key][source][cell] = batch_dprimes[func_key][source][site][..., cc, :]

            del batch_dprimes[func_key][source][site]

# set defaultdict factory functions as None to allow pickling
for middle_dict in batch_dprimes.values():
    middle_dict.default_factory = None
batch_dprimes.default_factory = None

# caches the bulk dprimes
batch_dprime_file = pl.Path(config['paths']['analysis_cache']) / 'prm_dprimes_v2' / set_name(meta)
if batch_dprime_file.parent.exists() is False:
    batch_dprime_file.parent.mkdir()

_ = dump(batch_dprimes, batch_dprime_file)
print(f'cacheing batch dprimes to {batch_dprime_file}')
