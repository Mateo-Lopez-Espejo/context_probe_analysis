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

memory = Memory(str(pl.Path(config['paths']['analysis_cache']) / 'prm_dprimes'))


@memory.cache
def single_cell_dprimes(site, probe, meta):
    # recs = load(site, remote=True, rasterfs=meta['raster_fs'], recache=False)
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
    trialR, R, _ = cdPCA.format_raster(raster)
    trialR, R = trialR.squeeze(axis=3), R.squeeze(axis=2)  # squeezes out probe

    rep, chn, ctx, tme = trialR.shape

    trans_pairs = [f'{x}_{y}' for x, y in itt.combinations(meta['transitions'], 2)]

    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, condition_axis=2,
                                  flip=meta['dprime_absolute'])  # shape CellPair x Cell x Time

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle

    shuffled = list()
    # pbar = ProgressBar()
    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)
    for tp in trans_pairs:
        shuf_trialR = np.empty([meta['montecarlo'], rep, chn, 2, tme])
        shuf_trialR[:] = np.nan

        tran_idx = np.array([meta['transitions'].index(int(t)) for t in tp.split('_')])
        ctx_shuffle = trialR[:, :, tran_idx, :].copy()

        for rr in range(meta['montecarlo']):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)

        shuffled.append(cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=3,
                                             flip=meta['dprime_absolute']))

    shuffled = np.stack(shuffled, axis=1).squeeze(axis=0).swapaxes(0, 1)  # shape Montecarlo x ContextPair x Cell x Time

    return dprime, shuffled, None, goodcells


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
    Re, C, S, T = trialR.shape

    # calculates full dPCA. i.e. considering all 4 categories
    dPCA_projection, dPCA_transformation = cdPCA.fit_transform(R, trialR)
    dprime = cDP.pairwise_dprimes(dPCA_projection, observation_axis=0, condition_axis=1,
                                  flip=meta['dprime_absolute'])

    # calculates floor (ctx shuffle) and ceiling (simulated data)
    sim_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))
    # sim_dprime = None
    shuf_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))

    # ctx_shuffle = trialR.copy()
    shuf_projection = dPCA_projection.copy()
    rng = np.random.default_rng(42)
    for rr in range(meta['montecarlo']):
        # ceiling: simulates data, calculates dprimes
        sim_trial = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0),
                                     size=[Re, C, S, T])
        sim_projection = cdPCA.transform(sim_trial, dPCA_transformation)
        sim_dprime[rr, ...] = cDP.pairwise_dprimes(sim_projection, observation_axis=0, condition_axis=1,
                                                   flip=meta['dprime_absolute'])

        # # floor: shuffles context identity, calculates dprime
        # ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)
        # shuf_projection = cdPCA.transform(ctx_shuffle, dPCA_transformation)
        shuf_projection = shuffle(shuf_projection, shuffle_axis=1, indie_axis=0, rng=rng)
        shuf_dprime[rr, ...] = cDP.pairwise_dprimes(shuf_projection, observation_axis=0, condition_axis=1,
                                                    flip=meta['dprime_absolute'])

    # # test plots
    # import matplotlib.pyplot as plt
    # import src.visualization.fancy_plots as fplt
    # fig, axes = plt.subplots(3,10)
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
    #     _ = fplt._cint(t, shuf_dprime[:, pair_idx, :], confidence=0.95, ax=axes[1,tt],
    #                    fillkwargs={'color': 'black', 'alpha': 0.5})
    #
    #     # plots the real dprime and simulatede dprime
    #     axes[2, tt].plot(t, dprime[pair_idx, :], color='black')
    #     _ = fplt._cint(t, sim_dprime[:, pair_idx, :], confidence=0.95, ax=axes[2, tt],
    #                    fillkwargs={'color': 'black', 'alpha': 0.5})

    return dprime, shuf_dprime, sim_dprime, goodcells


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

analysis_functions = {'SC': single_cell_dprimes, 'pdPCA': probewise_dPCA_dprimes}
# initilizede nested dictionary with three layer: 1. Analysis type 2. calculated values 3. cell or site
batch_dprimes = col.defaultdict(lambda: col.defaultdict(dict))
for site, (func_key, func) in itt.product(sites, analysis_functions.items()):

    real_dprimes = list()
    shuffled_dprimes = list()
    simulated_dprimes = list()
    shuffled_pvalues = list()
    simulated_pvalues = list()

    pvalues = col.defaultdict(list)

    for pp, probe in enumerate(all_probes):

        dprime, shuf_dprime, sim_dprime, cell_names = func(site, probe, meta)

        real_dprimes.append(dprime)
        shuffled_dprimes.append(shuf_dprime)
        simulated_dprimes.append(sim_dprime)

        # p value calculations
        for mont_vals, mont_name in zip([shuf_dprime, sim_dprime], ['shuffled', 'simulated']):

            if mont_vals is None:
                # catche the nonexistant simulated dprimes for single cell analysis
                continue

            if two_tail_p is True:
                top_pval = np.sum((mont_vals >= dprime), axis=0) / meta['montecarlo']
                bottom_pval = np.sum((mont_vals <= dprime), axis=0) / meta['montecarlo']
                dPCA_pvalues = np.where(dprime >= np.mean(mont_vals, axis=0), top_pval, bottom_pval)
                pvalues[mont_name].append(dPCA_pvalues)
            elif two_tail_p is False:
                dPCA_pvalues = np.sum((mont_vals >= dprime), axis=0) / meta['montecarlo']
                pvalues[mont_name].append(dPCA_pvalues)

    # appends probe arrays and  organizes into data structure
    batch_dprimes[func_key]['dprime'][site] = np.stack(real_dprimes)
    batch_dprimes[func_key]['shuffled_dprime'][site] = np.stack(shuffled_dprimes, axis=1)
    batch_dprimes[func_key]['shuffled_pvalue'][site] = np.stack(pvalues['shuffled'])
    if func_key != 'SC':
        batch_dprimes[func_key]['simulated_dprime'][site] = np.stack(simulated_dprimes, axis=1)
        batch_dprimes[func_key]['simulated_pvalue'][site] = np.stack(pvalues['simulated'])

    # for single cell analysis further divides the value by cell name
    if func_key == 'SC':
        for source in ['dprime', 'shuffled_dprime', 'shuffled_pvalue']:
            for (cc, cell) in enumerate(cell_names):
                batch_dprimes[func_key][source][cell] = batch_dprimes[func_key][source][site][..., cc, :]

            del batch_dprimes[func_key][source][site]

# set defaultdict factory functions as None to allow pickling
for middle_dict in batch_dprimes.values():
    middle_dict.default_factory = None
batch_dprimes.default_factory = None

# caches the bulk dprimes
batch_dprime_file = pl.Path(config['paths']['analysis_cache']) / 'prm_dprimes' / set_name(meta)
if batch_dprime_file.parent.exists() is False:
    batch_dprime_file.parent.mkdir()

_ = dump(batch_dprimes, batch_dprime_file)
print(f'cacheing batch dprimes to {batch_dprime_file}')
