import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation

import numpy as np
import pandas as pd
import scipy.stats as sst
import skimage.io as skio
from scipy.io import loadmat
import joblib as jl

import cpn_dPCA as cdPCA
import cpn_LDA as cLDA

import cpn_dprime as cDP
import fancy_plots
import fancy_plots as fplt
import fits as fts
from cpn_load import load
from cpp_cache import make_cache, get_cache, set_name
from reliability import signal_reliability
from tools import shuffle_along_axis as shuffle

"""
Summary of the d' context discrimination significance, and propulation effect significance across all combinations of 
sites and probes.
The two metrics extracted are the total number of significant time bins and the position of the last time bin.

it is highly recomended to add a way of keeping track of the distibution of significant bins over time across each
category
"""


def savefig(fig, root, name):
    root = pl.Path(config['paths']['figures']) / f'{root}'
    if not root.exists(): root.mkdir(parents=True, exist_ok=True)
    png = root.joinpath(name).with_suffix('.png')
    fancy_plots.savefig(png, transparent=False, dpi=100)
    # svg = root.joinpath(name).with_suffix('.svg')
    # fig.savefig(svg, transparent=True)


def bar_line(time, bar, line, ax=None, barkwargs={}, linekwargs={}):
    if ax is None:
        _, barax = plt.subplots()
    else:
        barax = ax

    lineax = barax.twinx()

    bar_defaults = {'color': 'C0'}
    for key, arg in bar_defaults.items(): barkwargs.setdefault(key, arg)
    line_defaults = {'color': 'C1'}
    for key, arg in line_defaults.items(): linekwargs.setdefault(key, arg)

    barax.bar(time, bar, **barkwargs)
    lineax.plot(time, line, **linekwargs)

    barax.tick_params(axis='y', labelcolor=barkwargs['color'])
    lineax.tick_params(axis='y', labelcolor=linekwargs['color'])

    return barax, lineax


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
    raster = cdPCA.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                   smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                   zscore=meta['zscore'], part='probe')

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    trialR, R = trialR.squeeze(axis=3), R.squeeze(axis=2)  # squeezes out probe

    rep, chn, ctx, tme = trialR.shape

    trans_pairs = [f'{x}_{y}' for x, y in itt.combinations(meta['transitions'], 2)]

    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, condition_axis=2, flip=meta['dprime_absolute'])  # shape CellPair x Cell x Time

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

    return dprime, shuffled, goodcells, trans_pairs


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
    raster = cdPCA.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                   smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                   zscore=meta['zscore'])

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, _, _ = cdPCA.format_raster(raster)
    trialR = trialR.squeeze()  # squeezes out probe
    Re, C, S, T = trialR.shape

    # calculates full LDA. i.e. considering all 4 categories
    LDA_projection, LDA_transformation = cLDA.fit_transform_over_time(trialR, 1)
    LDA_projection = LDA_projection.squeeze(axis=1)
    dprime = cDP.pairwise_dprimes(LDA_projection, observation_axis=0, condition_axis=1,
                                  flip=meta['dprime_absolute'])

    # calculates floor (ctx shuffle) and ceiling (simulated data)
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

    # test plots
    # fig, axes = plt.subplots(3,6)
    # t = np.arange(30)
    # for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
    #
    #     t0_idx = meta['transitions'].index(trans[0])
    #     t1_idx = meta['transitions'].index(trans[1])
    #
    #     axes[0,tt].plot(t, LDA_projection[:, t0_idx, :].mean(axis=0), color=trans_color_map[trans[0]], linewidth=3)
    #     axes[0,tt].plot(t, LDA_projection[:, t1_idx, :].mean(axis=0), color=trans_color_map[trans[1]], linewidth=3)
    #
    # # Raster, dprime, CI
    # bottom, top = axes[0, 0].get_ylim()
    # half = ((top - bottom) / 2) + bottom
    # for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
    #     pair_idx = SC_trans_pairs.index(f'{trans[0]}_{trans[1]}')
    #
    #     t0_idx = meta['transitions'].index(trans[0])
    #     t1_idx = meta['transitions'].index(trans[1])
    #
    #     _ = fplt._raster(t, LDA_projection[:, t0_idx, :], y_offset=0, y_range=(bottom, half), ax=axes[0,tt],
    #                      scatter_kws={'color': trans_color_map[trans[0]], 'alpha': 0.4, 's': 10})
    #     _ = fplt._raster(t, LDA_projection[:, t1_idx, :], y_offset=0, y_range=(half, top), ax=axes[0,tt],
    #                      scatter_kws={'color': trans_color_map[trans[1]], 'alpha': 0.4, 's': 10})
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
    raster = cdPCA.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
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


config = ConfigParser()
if pl.Path('../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../context_probe_analysis/config/settings.ini'))
elif pl.Path('../../../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../../../context_probe_analysis/config/settings.ini'))
else:
    raise FileNotFoundError('config file coluld not be foud')


trans_color_map = {'silence': '#377eb8',  # blue
                   'continuous': '#ff7f00',  # orange
                   'similar': '#4daf4a',  # green
                   'sharp': '#a65628'}  # brown

MC_color = {'shuffled': 'orange',
            'simulated': 'purple'}

# transferable plotting parameters
plt.rcParams['svg.fonttype'] = 'none'
sup_title_size = 30
sub_title_size = 20
ax_lab_size = 15
ax_val_size = 11
full_screen = [19.2, 9.83]
sns.set_style("ticks")

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute':None}


dprime_recache = False
rec_recache = False
two_tail_p = True

all_probes = [2, 3, 5, 6]

# sites = list(get_site_ids(316).keys())
sites = ['AMT028b', 'AMT029a', 'AMT030a', 'AMT031a', 'AMT032a', 'DRX008b', 'DRX021a', 'ley070a', 'ley072b']

region_map = dict(
    zip(['AMT028b', 'AMT029a', 'AMT030a', 'AMT031a', 'AMT032a', 'DRX008b', 'DRX021a', 'ley070a', 'ley072b'],
        ['PEG', 'PEG', 'PEG', 'PEG', 'PEG', 'A1', 'A1', 'A1', 'A1']))

SC_pvalues_dict = dict()
SC_reals_dict = dict()
SC_shuffled_dict = dict()

dPCA_pvalues_dict = dict()
dPCA_sim_pvalues_dict = dict()
dPCA_reals_dict = dict()
dPCA_shuffled_dict = dict()
dPCA_simulated_dict = dict()

LDA_pvalues_dict = dict()
LDA_sim_pvalues_dict = dict()
LDA_reals_dict = dict()
LDA_shuffled_dict = dict()
LDA_simulated_dict = dict()

# sites = [sites[5]]
for site in sites:

    this_site_SC_reals = list()
    this_site_SC_shuffled = list()
    this_site_SC_pvalues = list()

    this_site_dPCA_reals = list()
    this_site_dPCA_shuffled = list()
    this_site_dPCA_simulated = list()
    this_site_dPCA_pvalues = list()
    this_site_dPCA_sim_pvalues = list()

    this_site_LDA_reals = list()
    this_site_LDA_shuffled = list()
    this_site_LDA_simulated = list()
    this_site_LDA_pvalues = list()
    this_site_LDA_sim_pvalues = list()

    for pp, probe in enumerate(all_probes):
        ##############################
        # single cell analysis
        object_name = f'200221_{site}_P{probe}_single_cell_dprime'
        analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
        analysis_name = 'CPN_singel_cell_dprime'
        cache_folder = pl.Path(config['paths']['analysis_cache']) / f'{analysis_name}/{analysis_parameters}'

        SC_cache = make_cache(function=cell_dprime,
                              func_args={'site': site, 'probe': probe, 'meta': meta},
                              classobj_name=object_name,
                              cache_folder=cache_folder,
                              recache=dprime_recache)

        SC_dprime, SC_shuf_dprime, SC_cell_names, SC_trans_pairs = get_cache(SC_cache)

        this_site_SC_reals.append(SC_dprime)
        this_site_SC_shuffled.append(SC_shuf_dprime)

        #  p value base on the montecarlo shuffling
        if two_tail_p is True:
            top_pval = np.sum((SC_shuf_dprime >= SC_dprime), axis=0) / meta['montecarlo']
            bottom_pval = np.sum((SC_shuf_dprime <= SC_dprime), axis=0) / meta['montecarlo']
            SC_pvalues = np.where(SC_dprime >= np.mean(SC_shuf_dprime, axis=0), top_pval, bottom_pval)
            this_site_SC_pvalues.append(SC_pvalues)
        elif two_tail_p is False:
            SC_pvalues = np.sum((SC_shuf_dprime >= SC_dprime), axis=0) / meta['montecarlo']
            this_site_SC_pvalues.append(SC_pvalues)

        ##############################
        # dPCA analysis
        object_name = f'200221_{site}_P{probe}_single_cell_dprime'
        analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
        analysis_name = 'CPN_dPCA_dprime'
        cache_folder = pl.Path(config['paths']['analysis_cache']) / f'{analysis_name}/{analysis_parameters}'

        dPCA_cache = make_cache(function=dPCA_fourway_analysis,
                                func_args={'site': site, 'probe': probe, 'meta': meta},
                                classobj_name=object_name,
                                cache_folder=cache_folder,
                                recache=dprime_recache)
        dPCA_dprime, dPCA_shuf_dprime, dPCA_sim_dprime, dPCA_cell_names = get_cache(dPCA_cache)

        this_site_dPCA_reals.append(dPCA_dprime)
        this_site_dPCA_shuffled.append(dPCA_shuf_dprime)
        this_site_dPCA_simulated.append(dPCA_sim_dprime)

        #  p value base on the montecarlo shuffling
        if two_tail_p is True:
            top_pval = np.sum((dPCA_shuf_dprime >= dPCA_dprime), axis=0) / meta['montecarlo']
            bottom_pval = np.sum((dPCA_shuf_dprime <= dPCA_dprime), axis=0) / meta['montecarlo']
            dPCA_pvalues = np.where(dPCA_dprime >= np.mean(dPCA_shuf_dprime, axis=0), top_pval, bottom_pval)
            this_site_dPCA_pvalues.append(dPCA_pvalues)
        elif two_tail_p is False:
            dPCA_pvalues = np.sum((dPCA_shuf_dprime >= dPCA_dprime), axis=0) / meta['montecarlo']
            this_site_dPCA_pvalues.append(dPCA_pvalues)

        #  p value base on the montecarlo simulations
        if two_tail_p is True:
            top_pval = np.sum((dPCA_sim_dprime >= dPCA_dprime), axis=0) / meta['montecarlo']
            bottom_pval = np.sum((dPCA_sim_dprime <= dPCA_dprime), axis=0) / meta['montecarlo']
            dPCA_sim_pvalues = np.where(dPCA_dprime >= np.mean(dPCA_sim_dprime, axis=0), top_pval, bottom_pval)
            this_site_dPCA_sim_pvalues.append(dPCA_sim_pvalues)
        elif two_tail_p is False:
            dPCA_sim_pvalues = np.sum((dPCA_sim_dprime >= dPCA_dprime), axis=0) / meta['montecarlo']
            this_site_dPCA_sim_pvalues.append(dPCA_sim_pvalues)

        ##############################
        # LDA analysis
        object_name = f'200527_{site}_P{probe}_LDA_dprime'
        analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
        analysis_name = 'CPN_LDA_dprime'
        cache_folder = pl.Path(config['paths']['analysis_cache']) / f'{analysis_name}/{analysis_parameters}'
        LDA_cache = make_cache(function=LDA_fourway_analysis,
                             func_args={'site': site, 'probe': probe, 'meta': meta},
                             classobj_name=object_name,
                             cache_folder=cache_folder,
                             recache=dprime_recache)

        LDA_dprime, LDA_shuf_dprime, LDA_sim_dprime, LDA_cell_names = get_cache(LDA_cache)

        this_site_LDA_reals.append(LDA_dprime)
        this_site_LDA_shuffled.append(LDA_shuf_dprime)
        this_site_LDA_simulated.append(LDA_sim_dprime)

        #  p value base on the montecarlo shuffling
        if two_tail_p is True:
            top_pval = np.sum((LDA_shuf_dprime >= LDA_dprime), axis=0) / meta['montecarlo']
            bottom_pval = np.sum((LDA_shuf_dprime <= LDA_dprime), axis=0) / meta['montecarlo']
            LDA_pvalues = np.where(LDA_dprime >= np.mean(LDA_shuf_dprime, axis=0), top_pval, bottom_pval)
            this_site_LDA_pvalues.append(LDA_pvalues)
        elif two_tail_p is False:
            LDA_pvalues = np.sum((LDA_shuf_dprime >= LDA_dprime), axis=0) / meta['montecarlo']
            this_site_LDA_pvalues.append(LDA_pvalues)

        #  p value base on the montecarlo simulations
        if two_tail_p is True:
            top_pval = np.sum((LDA_sim_dprime >= LDA_dprime), axis=0) / meta['montecarlo']
            bottom_pval = np.sum((LDA_sim_dprime <= LDA_dprime), axis=0) / meta['montecarlo']
            LDA_sim_pvalues = np.where(LDA_dprime >= np.mean(LDA_sim_dprime, axis=0), top_pval, bottom_pval)
            this_site_LDA_sim_pvalues.append(LDA_sim_pvalues)
        elif two_tail_p is False:
            LDA_sim_pvalues = np.sum((LDA_sim_dprime >= LDA_dprime), axis=0) / meta['montecarlo']
            this_site_LDA_sim_pvalues.append(LDA_sim_pvalues)


    this_site_SC_reals = np.stack(this_site_SC_reals, axis=0)
    this_site_SC_shuffled = np.stack(this_site_SC_shuffled, axis=0)
    this_site_SC_pvalues = np.stack(this_site_SC_pvalues, axis=0)

    this_site_dPCA_reals = np.stack(this_site_dPCA_reals, axis=0)
    this_site_dPCA_shuffled = np.stack(this_site_dPCA_shuffled, axis=0)
    this_site_dPCA_simulated = np.stack(this_site_dPCA_simulated, axis=0)
    this_site_dPCA_pvalues = np.stack(this_site_dPCA_pvalues, axis=0)
    this_site_dPCA_sim_pvalues = np.stack(this_site_dPCA_sim_pvalues, axis=0)

    this_site_LDA_reals = np.stack(this_site_LDA_reals, axis=0)
    this_site_LDA_shuffled = np.stack(this_site_LDA_shuffled, axis=0)
    this_site_LDA_simulated = np.stack(this_site_LDA_simulated, axis=0)
    this_site_LDA_pvalues = np.stack(this_site_LDA_pvalues, axis=0)
    this_site_LDA_sim_pvalues = np.stack(this_site_LDA_sim_pvalues, axis=0)

    # reorders date in dictionary of cells
    for cc, cell in enumerate(SC_cell_names):
        SC_reals_dict[cell] = this_site_SC_reals[:, :, cc, :]
        SC_shuffled_dict[cell] = this_site_SC_shuffled[:, :, :, cc, :].swapaxes(0, 1)
        SC_pvalues_dict[cell] = this_site_SC_pvalues[:, :, cc, :]

    dPCA_reals_dict[site] = this_site_dPCA_reals
    dPCA_shuffled_dict[site] = this_site_dPCA_shuffled.swapaxes(0, 1)
    dPCA_simulated_dict[site] = this_site_dPCA_simulated.swapaxes(0, 1)
    dPCA_pvalues_dict[site] = this_site_dPCA_pvalues
    dPCA_sim_pvalues_dict[site] = this_site_dPCA_sim_pvalues

    LDA_reals_dict[site] = this_site_LDA_reals
    LDA_shuffled_dict[site] = this_site_LDA_shuffled.swapaxes(0, 1)
    LDA_simulated_dict[site] = this_site_LDA_simulated.swapaxes(0, 1)
    LDA_pvalues_dict[site] = this_site_LDA_pvalues
    LDA_sim_pvalues_dict[site] = this_site_LDA_sim_pvalues

########################################################################################################################
# defines arrays that identify cells, sites and regions
SC_cells_array = np.array(list(SC_pvalues_dict.keys()))
SC_sites_array = np.array([cell[0:7] for cell in SC_cells_array])
SC_regions_array = np.array(
    [region_map[cell[0:7]] for cell in SC_cells_array])

dPCA_site_array = np.array(list(dPCA_pvalues_dict.keys()))
dPCA_regions_array = np.array([cell[0:3] for cell in dPCA_site_array])

LDA_site_array = np.array(list(LDA_pvalues_dict.keys()))
LDA_regions_array = np.array([cell[0:3] for cell in LDA_site_array])

# defines a significatn threshold and transfroms the pvalues into bool (significant vs nonsignificant)
threshold = 0.01
SC_significance_dict = {key: (val <= threshold) for key, val in SC_pvalues_dict.items()}

dPCA_significance_dict = {key: (val <= threshold) for key, val in dPCA_pvalues_dict.items()}
dPCA_sim_significance_dict = {key: (val <= threshold) for key, val in dPCA_sim_pvalues_dict.items()}

LDA_significance_dict = {key: (val <= threshold) for key, val in LDA_pvalues_dict.items()}
LDA_sim_significance_dict = {key: (val <= threshold) for key, val in LDA_sim_pvalues_dict.items()}


# stacks arrays, with different time dimentions, padding with NAN
def nanstack(arr_dict):
    max_time = np.max([arr.shape[-1] for arr in arr_dict.values()])
    newdict = dict()
    for cell, arr in arr_dict.items():
        t = arr.shape[-1]
        if t < max_time:
            newshape = list(arr.shape[:-1])
            newshape.append(max_time)
            newarr = np.empty(newshape)
            newarr[:] = np.nan
            newarr[..., :t] = arr
        else:
            newarr = arr

        newdict[cell] = newarr

    stacked = np.stack(list(newdict.values()))
    return stacked


SC_reals_array = nanstack(SC_reals_dict)
SC_shuff_array = nanstack(SC_shuffled_dict).swapaxes(0, 1)  # swaps cells by monts
SC_significance_array = nanstack(SC_significance_dict)

dPCA_signif_array = nanstack(dPCA_significance_dict)
LDA_signif_array = nanstack(LDA_significance_dict)
########################################################################################################################

# set up the time bin labels in milliseconds, this is critical fro ploting and calculating the tau
nbin = SC_significance_array.shape[-1]
fs = meta['raster_fs']
times = np.linspace(0, nbin / fs, nbin, endpoint=False) * 1000

bar_width = 1 / fs * 1000
fig_root = 'single_cell_context_dprime'

########################################################################################################################
########################################################################################################################
########################################################################################################################
# plots all steps of analysis for example cell and site

def site_check_plot(site, probe):
    # loads the raw data
    recs = load(site, rasterfs=meta['raster_fs'], recache=rec_recache)
    sig = recs['trip0']['resp']
    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()
    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = cdPCA.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                   smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                   zscore=meta['zscore'], part='probe')
    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    trialR, R = trialR.squeeze(axis=3), R.squeeze(axis=2)  # squeezes out probe
    LDA, _ = cLDA.fit_transform_over_time(trialR)
    LDA = LDA.squeeze(axis=1)

    if meta['zscore'] is False:
        LDA = LDA*meta['raster_fs']

    # flips signs of dprimes and montecarlos as needed
    dprimes, shuffleds = cDP.flip_dprimes(LDA_reals_dict[site], LDA_shuffled_dict[site], flip='max')
    _, simulations = cDP.flip_dprimes(LDA_reals_dict[site], LDA_simulated_dict[site], flip='max')

    t = times[:LDA.shape[-1]]
    fig, axes = plt.subplots(3, 6, sharex='all', sharey='row')

    #  PSTH
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):

        t0_idx = meta['transitions'].index(trans[0])
        t1_idx = meta['transitions'].index(trans[1])

        axes[0, tt].plot(t, LDA[:, t0_idx, :].mean(axis=0), color=trans_color_map[trans[0]], linewidth=3)
        axes[0, tt].plot(t, LDA[:, t1_idx, :].mean(axis=0), color=trans_color_map[trans[1]], linewidth=3)

    # Raster, dprime, CI
    bottom, top = axes[0, 0].get_ylim()
    half = ((top - bottom) / 2) + bottom
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        prb_idx = all_probes.index(probe)
        pair_idx = SC_trans_pairs.index(f'{trans[0]}_{trans[1]}')

        t0_idx = meta['transitions'].index(trans[0])
        t1_idx = meta['transitions'].index(trans[1])

        # plots the real dprime and the shuffled dprime
        axes[1, tt].plot(t, dprimes[prb_idx, pair_idx, :], color='black')
        _ = fplt._cint(t, shuffleds[:, prb_idx, pair_idx, :], confidence=0.95, ax=axes[1, tt],
                       fillkwargs={'color': 'black', 'alpha': 0.5})

        # plots the real dprime and simulatede dprime
        axes[2, tt].plot(t, dprimes[prb_idx, pair_idx, :], color='black')
        _ = fplt._cint(t, simulations[:, prb_idx, pair_idx, :], confidence=0.95, ax=axes[2, tt],
                       fillkwargs={'color': 'black', 'alpha': 0.5})

    # significance bars
    ax1_bottom = axes[1, 0].get_ylim()[0]
    ax2_bottom = axes[2, 0].get_ylim()[0]
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        prb_idx = all_probes.index(probe)
        pair_idx = SC_trans_pairs.index(f'{trans[0]}_{trans[1]}')
        # context discrimination
        axes[1, tt].bar(t, LDA_significance_dict[site][prb_idx, pair_idx, :], width=bar_width, align='center',
                        edgecolor='white', bottom=ax1_bottom)

        # population effects
        axes[2, tt].bar(t, LDA_sim_significance_dict[site][prb_idx, pair_idx, :], width=bar_width, align='center',
                        edgecolor='white', bottom=ax2_bottom)

        # _ = fplt.exp_decay(t, SC_significance_dict[cell][prb_idx, pair_idx, :], ax=axes[2, tt])
        # if axes[2, tt].get_ylim()[1] < 1:
        #     axes[2, tt].set_ylim(0, 1)

        # formats legend
        if tt == 0:
            axes[0, tt].set_ylabel(f'LDA1', fontsize=ax_lab_size)
            axes[1, tt].set_ylabel(f'dprime', fontsize=ax_lab_size)
            axes[2, tt].set_ylabel(f'dprime', fontsize=ax_lab_size)
            axes[0, tt].tick_params(labelsize=ax_val_size)
            axes[1, tt].tick_params(labelsize=ax_val_size)
            axes[2, tt].tick_params(labelsize=ax_val_size)

        axes[2, tt].set_xlabel('time (ms)', fontsize=ax_lab_size)
        axes[2, tt].tick_params(labelsize=ax_val_size)
        axes[0, tt].set_title(f'{trans[0]}_{trans[1]}', fontsize=sub_title_size)

        for ax in np.ravel(axes):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    return fig, axes

for site in ['AMT028b', 'DRX008b']:
    site = 'DRX008b'
    probe = 2
    fig, axes = site_check_plot(site, probe=probe)
    half_screen = (full_screen[0], full_screen[1]/2)
    fig.set_size_inches(half_screen)
    title = f'LDA, {site} probe {probe}, calc steps'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, 'DAC3_figures', title)

########################################################################################################################
# summary plots for example cell and site

def site_summary_plot(site):
    # flips signs of dprimes and montecarlos as neede
    dprimes, shuffleds = cDP.flip_dprimes(LDA_reals_dict[site], LDA_shuffled_dict[site], flip='max')
    signif_bars =  LDA_significance_dict[site]

    t = times[:dprimes.shape[-1]]
    fig, axes = plt.subplots(5, 7, sharex='all', sharey='all')

    # dprime and confidence interval for each probe-transition combinations
    for (pp, probe), (tt, trans) in itt.product(enumerate(all_probes),
                                                enumerate(itt.combinations(meta['transitions'], 2))):
        prb_idx = all_probes.index(probe)
        pair_idx = SC_trans_pairs.index(f'{trans[0]}_{trans[1]}')

        # plots the real dprime and the shuffled dprime
        axes[pp, tt].plot(t, dprimes[prb_idx, pair_idx, :], color='black')
        _ = fplt._cint(t, shuffleds[:, prb_idx, pair_idx, :], confidence=0.95, ax=axes[pp, tt],
                       fillkwargs={'color': 'black', 'alpha': 0.5})
    # dprime and ci for the mean across context pairs
    for pp, probe in enumerate(all_probes):
        prb_idx = all_probes.index(probe)
        axes[pp, -1].plot(t, np.mean(dprimes[prb_idx, :, :], axis=0), color='black')
        axes[pp, -1].axhline(0, color='gray', linestyle='--')
        # _ = fplt._cint(t, np.mean(shuffleds[:, prb_idx, :, :], axis=1), confidence=0.95, ax=axes[pp, -1],
        #                fillkwargs={'color': 'black', 'alpha': 0.5})
    # dprime and ci for the mean across probes
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        pair_idx = SC_trans_pairs.index(f'{trans[0]}_{trans[1]}')
        axes[-1, tt].plot(t, np.mean(dprimes[:, pair_idx, :], axis=0), color='black')
        axes[-1, tt].axhline(0, color='gray', linestyle='--')
        # _ = fplt._cint(t, np.mean(shuffleds[:, :, pair_idx, :], axis=1), confidence=0.95, ax=axes[-1, tt],
        #                fillkwargs={'color': 'black', 'alpha': 0.5})


    # significance bars for each probe-transition combinations
    bar_bottom = axes[0, 0].get_ylim()[0]
    for (pp, probe), (tt, trans) in itt.product(enumerate(all_probes),
                                                enumerate(itt.combinations(meta['transitions'], 2))):
        prb_idx = all_probes.index(probe)
        pair_idx = SC_trans_pairs.index(f'{trans[0]}_{trans[1]}')
        axes[pp, tt].bar(t, signif_bars[prb_idx, pair_idx, :], width=bar_width, align='center',
                         edgecolor='white', bottom=bar_bottom)
        # _ = fplt.exp_decay(t, SC_significance_dict[cell][prb_idx, pair_idx, :], ax=axes[2, tt])
        # if axes[2, tt].get_ylim()[1] < 1:
        #     axes[2, tt].set_ylim(0, 1)
    # significance bars for the mean across context pairs
    for pp, probe in enumerate(all_probes):
        prb_idx = all_probes.index(probe)
        axes[pp, -1].bar(t, np.mean(signif_bars[prb_idx, :, :], axis=0), width=bar_width, align='center',
                         edgecolor='white', bottom=bar_bottom)
    # significance bars for the mean across probes
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        pair_idx = SC_trans_pairs.index(f'{trans[0]}_{trans[1]}')
        axes[-1, tt].bar(t, np.mean(signif_bars[:, pair_idx, :], axis=0), width=bar_width, align='center',
                         edgecolor='white', bottom=bar_bottom)


    # cell summary mean: dprime, confidence interval
    axes[-1, -1].plot(t, np.mean(dprimes[:, :, :], axis=(0,1)), color='black')
    axes[-1, -1].axhline(0, color='gray', linestyle='--')
    # _ = fplt._cint(t, np.mean(shuffleds[:, :, :, :], axis=(1,2)), confidence=0.95, ax=axes[-1, -1],
    #                fillkwargs={'color': 'black', 'alpha': 0.5})
    axes[-1, -1].bar(t, np.mean(signif_bars[:, :, :], axis=(0,1)), width=bar_width, align='center',
                     edgecolor='white', bottom=bar_bottom)


    # formats axis, legend and so on.
    for pp, probe in enumerate(all_probes):
        axes[pp, 0].set_ylabel(f'probe {probe}', fontsize=ax_lab_size)
        axes[pp, 0].tick_params(labelsize=ax_val_size)
    axes[-1, 0].set_ylabel(f'probe\nmean', fontsize=ax_lab_size)
    axes[-1, 0].tick_params(labelsize=ax_val_size)

    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        axes[0, tt].set_title(f'{trans[0]}_{trans[1]}', fontsize=sub_title_size)
        axes[-1, tt].set_xlabel('time (ms)', fontsize=ax_lab_size)
        axes[-1, tt].tick_params(labelsize=ax_val_size)
    axes[0, -1].set_title(f'pair\nmean', fontsize=sub_title_size)
    axes[-1, -1].set_xlabel('time (ms)', fontsize=ax_lab_size)
    axes[-1, -1].tick_params(labelsize=ax_val_size)

    for ax in np.ravel(axes):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axes

for site in ['AMT028b', 'DRX008b']:
    site = 'DRX008b'
    fig, axes = site_summary_plot(site)
    fig.set_size_inches(full_screen)
    title = f'LDA {site} probe pair summary'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, 'DAC3_figures', title)