import collections as col
import itertools as itt
import pathlib as pl
from configparser import ConfigParser
import joblib as jl

import numpy as np

import src.data.rasters
from src.data import LDA as cLDA, dPCA as cdPCA
from src.metrics import dprime as cDP
from src.data.load import load, get_site_ids
from src.data.cache import make_cache, get_cache, set_name
from src.metrics.reliability import signal_reliability
from src.utils.tools import shuffle_along_axis as shuffle

config = ConfigParser()
config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))


def cell_dprime(site, probe, meta):
    # recs = load(site, remote=True, rasterfs=meta['raster_fs'], recache=False)
    recs = load(site, rasterfs=meta['raster_fs'], recache=rec_recache)
    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
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
    for tp in trans_pairs:
        shuf_trialR = np.empty([meta['montecarlo'], rep, chn, 2, tme])
        shuf_trialR[:] = np.nan

        tran_idx = np.array([meta['transitions'].index(t) for t in tp.split('_')])
        ctx_shuffle = trialR[:, :, tran_idx, :].copy()

        for rr in range(meta['montecarlo']):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)

        shuffled.append(cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=3,
                                             flip=meta['dprime_absolute']))

    shuffled = np.stack(shuffled, axis=1).squeeze(axis=0).swapaxes(0, 1)  # shape Montecarlo x ContextPair x Cell x Time

    return dprime, shuffled, None, goodcells


def dPCA_fourway_analysis(site, probe, meta):
    recs = load(site, rasterfs=meta['raster_fs'], recache=rec_recache)

    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
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


def LDA_fourway_analysis(site, probe, meta):
    recs = load(site, rasterfs=meta['raster_fs'], recache=rec_recache)

    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
    sig = rec['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = src.data.rasters.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                              smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                              zscore=meta['zscore'])

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


meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None}

dprime_recache = True
rec_recache = True
two_tail_p = True

all_probes = [2, 3, 5, 6]
sites = set(get_site_ids(316).keys())
badsites = {'AMT031a', 'ley074a' } # empirically deciced
sites = sites.difference(badsites)

sites = ['CRD004a'] # test_site for debuging.

analysis_functions = {'SC': cell_dprime, 'dPCA': dPCA_fourway_analysis, 'LDA': LDA_fourway_analysis}
analysis_functions = {'dPCA': dPCA_fourway_analysis}
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

        # cache location and function name
        object_name = f'{site}_P{probe}_single_cell_dprime'
        analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
        analysis_name = f'CPN_{func_key}_dprime'
        cache_folder = pl.Path(config['paths']['analysis_cache']) / f'{analysis_name}/{analysis_parameters}'

        SC_cache = make_cache(function=func,
                              func_args={'site': site, 'probe': probe, 'meta': meta},
                              classobj_name=object_name,
                              cache_folder=cache_folder,
                              recache=dprime_recache)

        dprime, shuf_dprime, sim_dprime, cell_names = get_cache(SC_cache)

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

# set defaultdict factory functions as None to allowe pickling

for middle_dict in batch_dprimes.values():
    middle_dict.default_factory = None
batch_dprimes.default_factory = None


# caches the bulk dprimes
batch_dprime_file = pl.Path(config['paths']['analysis_cache']) / 'batch_dprimes' / set_name(meta)
if batch_dprime_file.parent.exists() is False:
    batch_dprime_file.parent.mkdir()

_ = jl.dump(batch_dprimes, batch_dprime_file)
print(f'cacheing batch dprimes to {batch_dprime_file}')
