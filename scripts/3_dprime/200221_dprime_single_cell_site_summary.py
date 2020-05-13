import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as skio
from scipy.io import loadmat

import cpn_dPCA as cdPCA
import cpn_dprime as cDP
import fancy_plots as fplt
import fits as fts
from cpn_load import load
from cpp_cache import make_cache, get_cache
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
    fig.savefig(png, transparent=False, dpi=100)
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


def dPCA_fourway_analysis(site, probe, meta):
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

    # test plots
    # fig, axes = plt.subplots(3,6)
    # t = np.arange(30)
    # for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
    #
    #     t0_idx = meta['transitions'].index(trans[0])
    #     t1_idx = meta['transitions'].index(trans[1])
    #
    #     axes[0,tt].plot(t, dPCA_projection[:, t0_idx, :].mean(axis=0), color=trans_color_map[trans[0]], linewidth=3)
    #     axes[0,tt].plot(t, dPCA_projection[:, t1_idx, :].mean(axis=0), color=trans_color_map[trans[1]], linewidth=3)
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
    #     _ = fplt._raster(t, dPCA_projection[:, t0_idx, :], y_offset=0, y_range=(bottom, half), ax=axes[0,tt],
    #                      scatter_kws={'color': trans_color_map[trans[0]], 'alpha': 0.4, 's': 10})
    #     _ = fplt._raster(t, dPCA_projection[:, t1_idx, :], y_offset=0, y_range=(half, top), ax=axes[0,tt],
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

bad_SC_sites = list()
SC_pvalues_dict = dict()
SC_reals_dict = dict()
SC_shuffled_dict = dict()

bad_dPCA_sites = list()
dPCA_pvalues_dict = dict()
dPCA_sim_pvalues_dict = dict()
dPCA_reals_dict = dict()
dPCA_shuffled_dict = dict()
dPCA_simulated_dict = dict()

for site in sites:

    this_site_SC_reals = list()
    this_site_SC_shuffled = list()
    this_site_SC_pvalues = list()

    this_site_dPCA_reals = list()
    this_site_dPCA_shuffled = list()
    this_site_dPCA_simulated = list()
    this_site_dPCA_pvalues = list()
    this_site_dPCA_sim_pvalues = list()

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

        # todo, eliminate test after everything works
        # pair_idx = 5 # 6 total
        # cell_idx = -2 # 79 total
        # dPCA_significance = dPCA_pvalues <= 0.05
        # barkwargs = dict(width=1, align='edge', color='black', edgecolor='white', alpha=0.5)
        # fig, ax = plt.subplots()
        # line = dPCA_dprime[pair_idx,:]
        # mont = dPCA_shuf_dprime[:,pair_idx,:]
        # hist = dPCA_significance[pair_idx,:]
        # np.arange(30)
        # ax.plot(np.arange(30), line, color='black')
        # _ = fplt._cint(np.arange(30), mont, confidence=0.95, ax=ax,
        #                fillkwargs={'color': 'blue', 'alpha': 0.5})
        # ax.bar(np.arange(30), hist, **barkwargs)

    this_site_SC_reals = np.stack(this_site_SC_reals, axis=0)
    this_site_SC_shuffled = np.stack(this_site_SC_shuffled, axis=0)
    this_site_SC_pvalues = np.stack(this_site_SC_pvalues, axis=0)

    this_site_dPCA_reals = np.stack(this_site_dPCA_reals, axis=0)
    this_site_dPCA_shuffled = np.stack(this_site_dPCA_shuffled, axis=0)
    this_site_dPCA_simulated = np.stack(this_site_dPCA_simulated, axis=0)
    this_site_dPCA_pvalues = np.stack(this_site_dPCA_pvalues, axis=0)
    this_site_dPCA_sim_pvalues = np.stack(this_site_dPCA_sim_pvalues, axis=0)

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

########################################################################################################################
# defines arrays that identify cells, sites and regions
SC_cells_array = np.array(list(SC_pvalues_dict.keys()))
SC_sites_array = np.array([cell[0:7] for cell in SC_cells_array])
SC_regions_array = np.array(
    [region_map[cell[0:7]] for cell in SC_cells_array])  # todo make a dictionary map from site to A1 or PEG

dPCA_site_array = np.array(list(dPCA_pvalues_dict.keys()))
dPCA_regions_array = np.array([cell[0:3] for cell in dPCA_site_array])

# defines a significatn threshold and transfroms the pvalues into bool (significant vs nonsignificant)
threshold = 0.01
SC_significance_dict = {key: (val <= threshold) for key, val in SC_pvalues_dict.items()}
dPCA_significance_dict = {key: (val <= threshold) for key, val in dPCA_pvalues_dict.items()}
dPCA_sim_significance_dict = {key: (val <= threshold) for key, val in dPCA_sim_pvalues_dict.items()}


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
#########################################################################################################################

# set up the time bin labels in milliseconds, this is critical fro ploting and calculating the tau
nbin = SC_significance_array.shape[-1]
fs = meta['raster_fs']
times = np.linspace(0, nbin / fs, nbin, endpoint=False) * 1000

bar_width = 1 / fs * 1000
fig_root = 'single_cell_context_dprime'

#########################################################################################################################
#########################################################################################################################
# dimensions to collapse per cell: Probe, Transition.
# collapse one, then the other, then both, per cell
# calculates the bins over/under a significant threshold

# for each cell in each site collapses across probes and context pairs
for site in sites:
    collapsed = np.nanmean(SC_significance_array[SC_sites_array == site, :, :, :], axis=(1, 2))
    site_cells = SC_cells_array[SC_sites_array == site]

    fig, axes = fplt.subplots_sqr(collapsed.shape[0], sharex=True, sharey=True, figsize=full_screen)
    for ax, hist, cell in zip(axes, collapsed, site_cells):
        ax.bar(times, hist, width=bar_width, align='edge', edgecolor='white')
        _ = fplt.exp_decay(times[:len(hist)], hist, ax=ax)
        ax.set_title(cell)
        ax.legend(loc='upper right')

    title = f'probes and transitions collapsed, single cell context dprime, {site}'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, fig_root, title)
    plt.close(fig)

# for each site collapeses across cells, rows are probes, columns are context pairs
site = 'DRX008b'
for site in set(SC_sites_array):
    site_mask = SC_sites_array == site
    arr = np.nanmean(SC_significance_array[site_mask, ...], axis=0)
    fig, axes = plt.subplots(arr.shape[0], arr.shape[1], sharex=True, sharey=True,
                             squeeze=False, figsize=full_screen)
    region = region_map[site[:7]]
    color = 'black' if region == 'A1' else 'red'

    for rr, (row, probe) in enumerate(zip(axes, all_probes)):
        for cc, (col, pair) in enumerate(zip(row, itt.combinations(meta['transitions'], 2))):
            ax = col
            line = np.nanmean(SC_reals_array[site_mask, ...], axis=0)[rr, cc, :]
            mont = np.nanmean(SC_shuff_array[:, site_mask, ...], axis=1)[:, rr, cc, :] #todo check that simiar lines were not wrong
            hist = arr[rr, cc, :]
            barkwargs = dict(width=bar_width, align='edge', color=color, edgecolor='white', alpha=0.5)
            linekwargs = dict(color='blue')

            ax.bar(times[:len(hist)], hist, **barkwargs)
            _ = fplt.exp_decay(times[:len(hist)], hist, ax=ax)

            ax.plot(times[:len(line)], line, **linekwargs)
            _ = fplt._cint(times[:mont.shape[1]], mont, confidence=0.95, ax=ax,
                           fillkwargs={'color': 'blue', 'alpha': 0.5})
            if cc == 0:
                ax.set_ylabel(f'probe{probe}')
            if rr == 0:
                ax.set_title(f'{pair[0]}_{pair[1]}')

            ax.legend()
    title = f'SC, transitions and probes comparison {site}'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, fig_root, title)
    plt.close(fig)

# for each site collapses across cells and probes
fig, axes = plt.subplots(len(sites),
                         len(list(itt.combinations(meta['transitions'], 2))),
                         sharex=True, sharey=True, squeeze=False, figsize=full_screen)
for row, site in enumerate(sites):
    region = region_map[site[:7]]
    color = 'black' if region == 'A1' else 'red'
    collapsed = np.nanmean(SC_significance_array[SC_sites_array == site, :, :, :], axis=(0, 1))
    for col, (ax, hist, pair) in enumerate(
            zip(axes[row], collapsed, itt.combinations(meta['transitions'], 2))):
        ax.bar(times, hist, width=bar_width, align='edge', color=color, edgecolor='white')
        _ = fplt.exp_decay(times, hist, ax=ax)
        ax.legend()

        if row == 0:
            ax.set_title(f'{pair[0]}_{pair[1]}')
        if col == 0:
            ax.set_ylabel(site)

title = f'cells and probes collapsed, transitions comparison'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
savefig(fig, fig_root, title)
plt.close(fig)

# for each site collapses across cells, probes, and transition pairs
fig, axes = plt.subplots(len(sites), 1,
                         sharex=True, sharey=True, squeeze=False, figsize=full_screen)
for row, (site, ax) in enumerate(zip(sites, axes)):
    ax = ax[0]
    region = region_map[site[:7]]
    color = 'black' if region == 'A1' else 'red'
    hist = np.nanmean(SC_significance_array[SC_sites_array == site, :, :, :], axis=(0, 1, 2))
    max_sig = hist.max()
    ax.bar(times, hist, width=bar_width, align='edge', color=color, edgecolor='white')
    _ = fplt.exp_decay(times, hist, ax=ax)
    ax.set_ylabel(site)
    ax.legend()
title = f'cells, probes and transitions collapsed, site mean dprime'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
savefig(fig, fig_root, title)
plt.close(fig)

# compares transitions: collapse across PEG vs A1, probe, cell
fig, axes = plt.subplots(2, 6, sharex=True, sharey=True, figsize=full_screen)
for row, region in enumerate(['A1', 'PEG']):
    color = 'black' if region == 'A1' else 'red'
    collapsed = np.nanmean(SC_significance_array[SC_regions_array == region, :, :, :], axis=(0, 1))

    for col, (ax, hist, pair) in enumerate(
            zip(axes[row], collapsed, itt.combinations(meta['transitions'], 2))):
        ax.bar(times, hist, width=bar_width, align='edge', color=color, edgecolor='white')
        _ = fplt.exp_decay(times, hist, ax=ax)
        ax.legend()
        if row == 0:
            ax.set_title(f'{pair[0]}_{pair[1]}')

        if col == 0:
            ax.set_ylabel(region)
title = f'cells, probes and region collapsed, region and transisiton comparision'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
savefig(fig, fig_root, title)
plt.close(fig)

#########################################################################################################################
#########################################################################################################################
# for each site dPCA plots the significant bins for each transition pair and probe
for site, arr in dPCA_significance_dict.items():
    fig, axes = plt.subplots(arr.shape[0], arr.shape[1], sharex=True, sharey=True,
                             squeeze=False, figsize=full_screen)
    region = region_map[site[:7]]
    color = 'black' if region == 'A1' else 'red'

    for rr, (row, probe) in enumerate(zip(axes, all_probes)):
        for cc, (col, pair) in enumerate(zip(row, itt.combinations(meta['transitions'], 2))):
            ax = col
            line = dPCA_reals_dict[site][rr, cc, :]
            mont = dPCA_shuffled_dict[site][:, rr, cc, :]
            hist = arr[rr, cc, :]
            ax.bar(times[:len(hist)], hist, width=bar_width, align='edge', color=color, edgecolor='white', alpha=0.5)
            _ = fplt.exp_decay(times[:len(hist)], hist, ax=ax)

            ax.plot(times[:len(line)], line, color='blue')
            _ = fplt._cint(times[:mont.shape[1]], mont, confidence=0.95, ax=ax,
                           fillkwargs={'color': 'blue', 'alpha': 0.5})
            if cc == 0:
                ax.set_ylabel(f'probe{probe}')
            if rr == 0:
                ax.set_title(f'{pair[0]}_{pair[1]}')

            ax.legend()
    title = f'dPCA, transitions and probes comparison {site}'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, fig_root, title)
    plt.close(fig)

# for each site dPCA, plots the significant bins for each transitions pair, collapses probes
fig, axes = plt.subplots(len(dPCA_significance_dict), arr.shape[1], sharex=True, sharey=True,
                         squeeze=False, figsize=full_screen)
for rr, (row, (site, arr)) in enumerate(zip(axes, dPCA_significance_dict.items())):
    region = region_map[site[:7]]
    color = 'black' if region == 'A1' else 'red'
    collapsed = np.nanmean(arr, axis=0)
    for cc, (col, pair) in enumerate(zip(row, itt.combinations(meta['transitions'], 2))):
        ax = col
        hist = collapsed[cc, :]
        ax.bar(times[:len(hist)], hist, width=bar_width, align='edge', color=color, edgecolor='white')
        _ = fplt.exp_decay(times[:len(hist)], hist, ax=ax)

        if cc == 0:
            ax.set_ylabel(f'{site}\nmean significant bins')
        if rr == 0:
            ax.set_title(f'{pair[0]}_{pair[1]}')

        ax.legend()

title = f'collapsed probes and units, transitions comparision'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
savefig(fig, fig_root, title)
plt.close(fig)

# for each site collapses across cells, probes, and transition pairs, compares single cell with dPCA
fig, axes = plt.subplots(len(sites), 2, sharex=True, sharey=True,
                         squeeze=False, figsize=full_screen)
for row, (site, ax) in enumerate(zip(sites, axes)):
    color = 'red' if site[0:3] == 'AMT' else 'black'
    SC_collapsed = np.nanmean(SC_significance_array[SC_sites_array == site, :, :, :], axis=(0, 1, 2))
    ax[0].bar(times, SC_collapsed, width=bar_width, align='edge', color=color, edgecolor='white')
    _ = fplt.exp_decay(times[:len(SC_collapsed)], SC_collapsed, ax=ax[0])
    ax[0].set_ylabel(site)
    ax[0].set_title('mean of single cells')
    ax[0].legend()

    dPCA_collapsed = np.nanmean(dPCA_significance_dict[site], axis=(0, 1))
    ax[1].bar(times[:len(dPCA_collapsed)], dPCA_collapsed, width=bar_width, align='edge', color=color,
              edgecolor='white')
    _ = fplt.exp_decay(times, dPCA_collapsed, ax=ax[1])
    ax[1].set_title('dPCA')
    ax[1].legend()

title = 'cells, probes and transitions collapsed, single cell vs dPCA significant integration bins'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
savefig(fig, fig_root, title)
plt.close(fig)

#########################################################################################################################
#########################################################################################################################
# data frame containing all the important summary data, i.e. exponential decay fits for dprime and significance, for
# all combinantions of transition parirs, and probes,  for the means across probes, transistion pairs or for both, and
# for the single cell analysis or the dPCA projections
df = list()
for site in sites:
    dPCA_reals_dict[site]
    dPCA_significance_dict[site]

    dprimes, _ = cDP.flip_dprimes(dPCA_reals_dict[site], None, flip='max')

    site_cells = SC_cells_array[SC_sites_array == site]
    for cells in site_cells[site]:


    collapsed = np.nanmean(SC_significance_array[SC_sites_array == site, :, :, :], axis=(1, 2))
    site_popt, _ = fts.exp_decay(times, np.nanmean(collapsed, axis=0))
    dPCA_popt, _ = fts.exp_decay(times, np.nanmean(dPCA_significance_dict[site], axis=(0, 1)))
    site_cells = SC_cells_array[SC_sites_array == site]
    for cell, hist in zip(site_cells, collapsed):
        cell_popt, _ = fts.exp_decay(times, hist)
        for sname, source in zip(('mean', 'dPCA', 'cell'), (site_popt, dPCA_popt, cell_popt)):
            for parameter, value in zip(('r0', 'decay'), source):
                d = {'siteid': site,
                     'cellid': cell,
                     'analysis': analysis, # singel_cell, dPCA
                     'probe': probe, # probe_n, mean
                     'transition_pair': transition_pair # t0_t1, mean
                     'parameter': parameter,
                     'value': value}
                df.append(d)
DF = pd.DataFrame(df)




df = list()
for site in sites:
    collapsed = np.nanmean(SC_significance_array[SC_sites_array == site, :, :, :], axis=(1, 2))
    site_popt, _ = fts.exp_decay(times, np.nanmean(collapsed, axis=0))
    dPCA_popt, _ = fts.exp_decay(times, np.nanmean(dPCA_significance_dict[site], axis=(0, 1)))
    site_cells = SC_cells_array[SC_sites_array == site]
    for cell, hist in zip(site_cells, collapsed):
        cell_popt, _ = fts.exp_decay(times, hist)
        for sname, source in zip(('mean', 'dPCA', 'cell'), (site_popt, dPCA_popt, cell_popt)):
            for parameter, value in zip(('r0', 'decay'), source):
                d = {'siteid': site,
                     'cellid': cell,
                     'source': sname,
                     'parameter': parameter,
                     'value': value}
                df.append(d)
DF = pd.DataFrame(df)
# add tau defined as the inverse of the decay constant
decay = DF.loc[DF.parameter == 'decay', :].copy()
decay['parameter'] = 'tau'
decay['value'] = -1 / decay['value']
DF = pd.concat((DF, decay), axis=0)

ff_param = DF.parameter == 'tau'
ff_threshold = DF.value < 1000  # drop cells with abnormally high levels of tau, i.e, over 1 second
filtered = DF.loc[ff_param & ff_threshold, :]
pivoted = filtered.pivot(index='cellid', columns='source', values='value').dropna()

# plots the raw dPCA vs single Cell values without any further filtering besides gross artifacts like taus > 1s
fig, ax = plt.subplots(figsize=full_screen)
y = pivoted['dPCA']
x = pivoted['cell']
ax.scatter(x, y)
_ = fplt.lin_reg(x, y, ax=ax)
ax.set_xlabel(x.name)
ax.set_ylabel(y.name)
ax.legend()
title = 'dPCA vs sigle cell comparison'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
# savefig(fig, fig_root, title)

#########################################################################################################################
#########################################################################################################################
# plots all steps of analysis for example cell and site
def cell_check_plot(cell, probe):
    site = cell[:7]

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
    trialR = trialR.squeeze()
    if meta['zscore'] is False:
        trialR = trialR*meta['raster_fs']

    # flips signs of dprimes and montecarlos as neede
    dprimes, shuffleds = cDP.flip_dprimes(SC_reals_dict[cell], SC_shuffled_dict[cell], flip='max')

    t = times[:trialR.shape[-1]]
    fig, axes = plt.subplots(2, 6, sharex='all', sharey='row')

    #  PSTH
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        cell_idx = goodcells.index(cell)

        t0_idx = meta['transitions'].index(trans[0])
        t1_idx = meta['transitions'].index(trans[1])

        axes[0, tt].plot(t, trialR[:, cell_idx, t0_idx, :].mean(axis=0), color=trans_color_map[trans[0]], linewidth=3)
        axes[0, tt].plot(t, trialR[:, cell_idx, t1_idx, :].mean(axis=0), color=trans_color_map[trans[1]], linewidth=3)

    # Raster, dprime, CI
    bottom, top = axes[0, 0].get_ylim()
    half = ((top - bottom) / 2) + bottom
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        cell_idx = goodcells.index(cell)
        prb_idx = all_probes.index(probe)
        pair_idx = SC_trans_pairs.index(f'{trans[0]}_{trans[1]}')

        t0_idx = meta['transitions'].index(trans[0])
        t1_idx = meta['transitions'].index(trans[1])

        _ = fplt._raster(t, trialR[:, cell_idx, t0_idx, :], y_offset=0, y_range=(bottom, half), ax=axes[0, tt],
                         scatter_kws={'color': trans_color_map[trans[0]], 'alpha': 0.4, 's': 10})
        _ = fplt._raster(t, trialR[:, cell_idx, t1_idx, :], y_offset=0, y_range=(half, top), ax=axes[0, tt],
                         scatter_kws={'color': trans_color_map[trans[1]], 'alpha': 0.4, 's': 10})

        # plots the real dprime and the shuffled dprime
        axes[1, tt].plot(t, dprimes[prb_idx, pair_idx, :], color='black')
        # axes[1].plot(all_shuffled[cell][:, prb_idx, pair_idx, :].T, color='green', alpha=0.01)
        _ = fplt._cint(t, shuffleds[:, prb_idx, pair_idx, :], confidence=0.95, ax=axes[1, tt],
                       fillkwargs={'color': 'black', 'alpha': 0.5})

    # significance bars
    bar_bottom = axes[1, 0].get_ylim()[0]
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        prb_idx = all_probes.index(probe)
        pair_idx = SC_trans_pairs.index(f'{trans[0]}_{trans[1]}')
        # plots the histogram of significant bins
        axes[1, tt].bar(t, SC_significance_dict[cell][prb_idx, pair_idx, :], width=bar_width, align='center',
                        edgecolor='white', bottom=bar_bottom)
        # _ = fplt.exp_decay(t, SC_significance_dict[cell][prb_idx, pair_idx, :], ax=axes[2, tt])
        # if axes[2, tt].get_ylim()[1] < 1:
        #     axes[2, tt].set_ylim(0, 1)

        # formats legend
        if tt == 0:
            axes[0, tt].set_ylabel(f'spike rate\nz-score', fontsize=ax_lab_size)
            axes[1, tt].set_ylabel(f'dprime', fontsize=ax_lab_size)
            axes[0, tt].tick_params(labelsize=ax_val_size)
            axes[1, tt].tick_params(labelsize=ax_val_size)
            # axes[2, tt].set_ylabel(f'significant bins')
        axes[1, tt].set_xlabel('time (ms)', fontsize=ax_lab_size)
        axes[1, tt].tick_params(labelsize=ax_val_size)
        axes[0, tt].set_title(f'{trans[0]}_{trans[1]}', fontsize=sub_title_size)

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    return fig, axes

cell = 'AMT029a-57-1'
cell = 'DRX021a-10-2'
cell = 'DRX008b-99-7'
probe = 6
fig, axes = cell_check_plot(cell, probe=probe)
half_screen = (full_screen[0], full_screen[1]/2)
fig.set_size_inches(half_screen)
title = f'{cell} probe {probe} calc steps'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
# savefig(fig, fig_root, title)

def site_check_plot(site, probe):

    # loads the raw data
    # recs = load(site)
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
    dPCs, _ = cdPCA.fit_transform(R, trialR)

    if meta['zscore'] is False:
        dPCs = dPCs*meta['raster_fs']

    # flips signs of dprimes and montecarlos as needed
    dprimes, shuffleds = cDP.flip_dprimes(dPCA_reals_dict[site], dPCA_shuffled_dict[site], flip='max')
    _, simulations = cDP.flip_dprimes(dPCA_reals_dict[site], dPCA_simulated_dict[site], flip='max')

    t = times[:dPCs.shape[-1]]
    fig, axes = plt.subplots(3, 6, sharex='all', sharey='row')

    #  PSTH
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):

        t0_idx = meta['transitions'].index(trans[0])
        t1_idx = meta['transitions'].index(trans[1])

        axes[0, tt].plot(t, dPCs[:, t0_idx, :].mean(axis=0), color=trans_color_map[trans[0]], linewidth=3)
        axes[0, tt].plot(t, dPCs[:, t1_idx, :].mean(axis=0), color=trans_color_map[trans[1]], linewidth=3)

    # Raster, dprime, CI
    bottom, top = axes[0, 0].get_ylim()
    half = ((top - bottom) / 2) + bottom
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        prb_idx = all_probes.index(probe)
        pair_idx = SC_trans_pairs.index(f'{trans[0]}_{trans[1]}')

        t0_idx = meta['transitions'].index(trans[0])
        t1_idx = meta['transitions'].index(trans[1])

        _ = fplt._raster(t, dPCs[:, t0_idx, :], y_offset=0, y_range=(bottom, half), ax=axes[0, tt],
                         scatter_kws={'color': trans_color_map[trans[0]], 'alpha': 0.4, 's': 10})
        _ = fplt._raster(t, dPCs[:, t1_idx, :], y_offset=0, y_range=(half, top), ax=axes[0, tt],
                         scatter_kws={'color': trans_color_map[trans[1]], 'alpha': 0.4, 's': 10})

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
        axes[1, tt].bar(t, dPCA_significance_dict[site][prb_idx, pair_idx, :], width=bar_width, align='center',
                        edgecolor='white', bottom=ax1_bottom)

        # population effects
        axes[2, tt].bar(t, dPCA_sim_significance_dict[site][prb_idx, pair_idx, :], width=bar_width, align='center',
                        edgecolor='white', bottom=ax2_bottom)

        # _ = fplt.exp_decay(t, SC_significance_dict[cell][prb_idx, pair_idx, :], ax=axes[2, tt])
        # if axes[2, tt].get_ylim()[1] < 1:
        #     axes[2, tt].set_ylim(0, 1)

        # formats legend
        if tt == 0:
            axes[0, tt].set_ylabel(f'dPC', fontsize=ax_lab_size)
            axes[1, tt].set_ylabel(f'dprime', fontsize=ax_lab_size)
            axes[2, tt].set_ylabel(f'dprime', fontsize=ax_lab_size)
            axes[0, tt].tick_params(labelsize=ax_val_size)
            axes[1, tt].tick_params(labelsize=ax_val_size)
            axes[2, tt].tick_params(labelsize=ax_val_size)

        axes[2, tt].set_xlabel('time (ms)', fontsize=ax_lab_size)
        axes[2, tt].tick_params(labelsize=ax_val_size)
        axes[0, tt].set_title(f'{trans[0]}_{trans[1]}', fontsize=sub_title_size)

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    return fig, axes
site = 'AMT029a'
site = 'DRX021a'
site = 'DRX008b'
probe = 6
fig, axes = site_check_plot(site, probe=probe)
half_screen = (full_screen[0], full_screen[1]/2)
fig.set_size_inches(half_screen)
title = f'{site} probe {probe}, calc steps'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
# savefig(fig, fig_root, title)
#########################################################################################################################
# sumamry plots for example cell and site

def cell_summary_plot(cell):
    # flips signs of dprimes and montecarlos as neede
    dprimes, shuffleds = cDP.flip_dprimes(SC_reals_dict[cell], SC_shuffled_dict[cell], flip='max')
    signif_bars =  SC_significance_dict[cell]

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

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axes
cell = 'DRX008b-99-7'
fig, axes = cell_summary_plot(cell)
fig.set_size_inches(full_screen)
title = f'{cell} probe pair summary'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
# savefig(fig, fig_root, title)

def site_summary_plot(site):
    # flips signs of dprimes and montecarlos as neede
    dprimes, shuffleds = cDP.flip_dprimes(dPCA_reals_dict[site], dPCA_shuffled_dict[site], flip='max')
    signif_bars =  dPCA_significance_dict[site]

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

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axes
site = 'DRX008b'
fig, axes = site_summary_plot(site)
fig.set_size_inches(full_screen)
title = f'{site} probe pair summary'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
# savefig(fig, fig_root, title)

#########################################################################################################################
# fit and metrics example
def cell_fit_plot(cell):
    # flips signs of dprimes and montecarlos as neede
    dprimes, shuffleds = cDP.flip_dprimes(SC_reals_dict[cell], SC_shuffled_dict[cell], flip='max')
    signif_bars =  SC_significance_dict[cell]

    mean_dprime = np.mean(dprimes[:, :, :], axis=(0,1))
    mean_signif = np.mean(signif_bars[:, :, :], axis=(0,1))

    t = times[:dprimes.shape[-1]]
    fig, axes = plt.subplots(2, 1, sharex='all', sharey='all')

    # plots dprime plus fit
    axes[0].plot(t, mean_dprime, color='black')
    axes[0].axhline(0, color='gray', linestyle='--')
    _ = fplt.exp_decay(t, mean_dprime, ax=axes[0], linestyle='--', color='black')

    # plots confifence bins plut fit
    axes[1].bar(t, mean_signif, width=bar_width, align='center',
                     edgecolor='white',)
    _ = fplt.exp_decay(times, mean_signif, ax=axes[1], linestyle='--', color='black')

    axes[0].legend(loc='upper right', fontsize=ax_val_size, markerscale=3, frameon=False, )
    axes[1].legend(loc='upper right', fontsize=ax_val_size, markerscale=3, frameon=False, )

    # formats axis, legend and so on.

    axes[0].set_ylabel(f'dprime', fontsize=ax_lab_size)
    axes[0].tick_params(labelsize=ax_val_size)
    axes[1].set_ylabel(f'mean significance', fontsize=ax_lab_size)
    axes[1].tick_params(labelsize=ax_val_size)

    axes[1].set_xlabel('time (ms)', fontsize=ax_lab_size)
    axes[1].tick_params(labelsize=ax_val_size)

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axes
cell = 'DRX008b-99-7'
fig, axes = cell_fit_plot(cell)
fig.set_size_inches((8,8))
title = f'{cell} fit summary'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
# savefig(fig, fig_root, title)

def site_fit_plot(site):
    # flips signs of dprimes and montecarlos as neede
    dprimes, shuffleds = cDP.flip_dprimes(dPCA_reals_dict[site], dPCA_shuffled_dict[site], flip='max')
    signif_bars =  dPCA_significance_dict[site]

    mean_dprime = np.mean(dprimes[:, :, :], axis=(0,1))
    mean_signif = np.mean(signif_bars[:, :, :], axis=(0,1))

    t = times[:dprimes.shape[-1]]
    fig, axes = plt.subplots(2, 1, sharex='all', sharey='all')

    # plots dprime plus fit
    axes[0].plot(t, mean_dprime, color='black')
    axes[0].axhline(0, color='gray', linestyle='--')
    _ = fplt.exp_decay(t, mean_dprime, ax=axes[0], linestyle='--', color='black')

    # plots confifence bins plut fit
    axes[1].bar(t, mean_signif, width=bar_width, align='center',
                edgecolor='white',)
    _ = fplt.exp_decay(times, mean_signif, ax=axes[1], linestyle='--', color='black')

    axes[0].legend(loc='upper right', fontsize=ax_val_size, markerscale=3, frameon=False, )
    axes[1].legend(loc='upper right', fontsize=ax_val_size, markerscale=3, frameon=False, )

    # formats axis, legend and so on.

    axes[0].set_ylabel(f'dprime', fontsize=ax_lab_size)
    axes[0].tick_params(labelsize=ax_val_size)
    axes[1].set_ylabel(f'mean significance', fontsize=ax_lab_size)
    axes[1].tick_params(labelsize=ax_val_size)

    axes[1].set_xlabel('time (ms)', fontsize=ax_lab_size)
    axes[1].tick_params(labelsize=ax_val_size)

    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axes
site = 'DRX008b'
fig, axes = site_fit_plot(site)
fig.set_size_inches((8,8))
title = f'{site} fit summary'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
# savefig(fig, fig_root, title)

#########################################################################################################################
#########################################################################################################################
def compare_plot(cell):
    site = cell[0:7]

    SC_hist = np.nanmean(SC_significance_dict[cell], axis=(0, 1))
    SC_dprime = np.nanmean(SC_reals_dict[cell], axis=(0, 1))

    dPCA_hist = np.nanmean(dPCA_significance_dict[site], axis=(0, 1))
    dPCA_dprime = np.nanmean(dPCA_reals_dict[site], axis=(0, 1))

    fig = plt.figure()
    gs = fig.add_gridspec(2, 3)

    barkwargs = dict(width=bar_width, align='edge', color='black', edgecolor='white')
    linekwargs = dict(color='blue')

    # single cell summary
    ax = fig.add_subplot(gs[0, 0])
    SC_barax, SC_lineax = bar_line(times[:len(SC_hist)], SC_hist, SC_dprime, ax=ax,
                                   barkwargs=barkwargs, linekwargs=linekwargs)
    _ = fplt.exp_decay(times[:len(SC_hist)], SC_hist, ax=SC_barax)
    SC_barax.legend()

    # dPCA for this cell site
    ax = fig.add_subplot(gs[1, 0])
    dPCA_barax, dPCA_lineax = bar_line(times[:len(dPCA_hist)], dPCA_hist, dPCA_dprime, ax=ax,
                                       barkwargs=barkwargs, linekwargs=linekwargs)
    _ = fplt.exp_decay(times[:len(dPCA_hist)], dPCA_hist, ax=dPCA_barax)
    dPCA_barax.legend()

    # share axes, format axes names
    SC_barax.get_shared_y_axes().join(SC_barax, dPCA_barax)
    SC_lineax.get_shared_y_axes().join(SC_lineax, dPCA_lineax)
    SC_barax.get_shared_x_axes().join(SC_barax, SC_lineax, dPCA_barax, dPCA_lineax)

    SC_barax.set_title('Single cell')
    dPCA_barax.set_title('site dPCA')

    SC_barax.set_ylabel('mean significant bins')
    SC_lineax.set_ylabel('mean d-prime')

    dPCA_barax.set_ylabel('mean significant bins')
    dPCA_lineax.set_ylabel('mean d-prime')
    dPCA_barax.set_xlabel('probe time (ms)')

    # find the PNGs of sam analysis and add to the figure
    samm_fig_dir = pl.Path(config['paths']['sam_analysis']) / 'figures/lag-correlation/'

    corr_path = list(samm_fig_dir.glob(f'{site}*\\*\\{cell}-win*-range*.png'))[0]
    model_path = list(samm_fig_dir.glob(f'{site}*\\*\\{cell}-model-prediction-lineplot_*.png'))[0]

    sam_corr = skio.imread(corr_path)
    sam_model = skio.imread(model_path)

    corr_ax = fig.add_subplot(gs[0, 1:])
    model_ax = fig.add_subplot(gs[1, 1:])

    # fig, (corr_ax, model_ax) = plt.subplots(2,1)

    corr_ax.imshow(sam_corr[30:570, 50:720, :])
    model_ax.imshow(sam_model[30:570, 50:720, :])

    for aa in (corr_ax, model_ax):
        aa.get_xaxis().set_visible(False)
        aa.get_yaxis().set_visible(False)
        for ll in ['top', 'bottom', 'right', 'left']:
            aa.spines[ll].set_visible(False)

    corr_ax.set_title('lag correlation')
    model_ax.set_title('model fit')

    return fig

# get list of cells with sams analysisi
file = pl.Path(config['paths']['sam_analysis']) / 'analysis/model_fit_pop_summary.mat'
best_fits = loadmat(file)['best_fits'].squeeze()
# orders the data in DF
df = list()
for row in best_fits:
    df.append({'cellid': row[2][0],
               'intper_ms': row[0][0][0],
               'delay_ms': row[1][0][0]})
integration_fits = pd.DataFrame(df)
integration_cells = set(integration_fits['cellid'].unique())
context_cells = set(SC_cells_array)
common_cells = context_cells.intersection(integration_cells)

# plots the comparison figures
cell = 'DRX021a-10-2'
cell = 'DRX008b-102-4'
for cell in common_cells:
    fig = compare_plot(cell)
    title = f'{cell} context vs integration'
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.suptitle(title)
    fig.set_size_inches([10.13, 9.74])
    savefig(fig, 'single_cell_comparison', title)
    plt.close(fig)


########################################################################################################################
# considering the d' itself, and tau fitted to it
def fit_line(time, value):
    popt, _ = fts.exp_decay(time[:len(value)], value)
    line = fts._exp(time[:len(value)], *popt)
    return line

# first single cell comparisons, fit of the means of mean of the fits ...
cell = 'DRX008b-99-7'
for cell in common_cells:
    fig_folder = 'cell_dprime_signif_fits'
    title = f'{cell}_dprime_signif_fits'
    fig_file = pl.Path(config['paths']['figures']) / f'{fig_folder}/{title}.png'
    if fig_file.exists():
        print(f'{cell} figure alredy exists')
        continue
    else:
        print(f'{cell} creating figure')

    # plots cell mean significance and dprime
    fig, axes = plt.subplots(1, 5, figsize=full_screen)
    signif = np.mean(SC_significance_dict[cell], (0, 1))
    dprime = np.mean(SC_reals_dict[cell], (0, 1))
    barkwargs = dict(width=bar_width, align='edge', color='black', edgecolor='white')
    linekwargs = dict(color='blue')
    barax, lineax = bar_line(times[:len(signif)], signif, dprime, ax=axes[0],
                             barkwargs=barkwargs, linekwargs=linekwargs)
    barax.set_title('full mean of values')

    # 1.mean probe and transition, 2 fit
    try:
        s_fit = fit_line(times[:len(signif)], signif)
        d_fit = fit_line(times[:len(dprime)], dprime)
        axes[1].plot(times[:len(s_fit)], s_fit, color='black')
        twinx1 = axes[1].twinx()
        twinx1.plot(times[:len(d_fit)], d_fit, color='blue')
        axes[1].set_title('fit(mean(transitions x probes))')
    except:
        pass

    # 1. mean of probes 2. fit of transitions 3. mean of transitions
    try:
        signif = np.mean(SC_significance_dict[cell], (0))
        dprime = np.mean(SC_reals_dict[cell], (0))
        s_fit = np.mean(np.stack([fit_line(times[:len(s)], s) for s in signif]), 0)
        d_fit = np.mean(np.stack([fit_line(times[:len(d)], d) for d in dprime]), 0)
        axes[2].plot(times[:len(s_fit)], s_fit, color='black')
        twinx2 = axes[2].twinx()
        twinx2.plot(times[:len(d_fit)], d_fit, color='blue')
        axes[2].set_title('mean(fit(transitions x mean(probes)))')
    except:
        pass

    # 1. mean of transitions 2. fit of probes 3. mean of transitions
    try:
        signif = np.mean(SC_significance_dict[cell], (1))
        dprime = np.mean(SC_reals_dict[cell], (1))
        s_fit = np.mean(np.stack([fit_line(times[:len(s)], s) for s in signif]), 0)
        d_fit = np.mean(np.stack([fit_line(times[:len(d)], d) for d in dprime]), 0)
        axes[3].plot(times[:len(s_fit)], s_fit, color='black')
        twinx3 = axes[3].twinx()
        twinx3.plot(times[:len(d_fit)], d_fit, color='blue')
        axes[3].set_title('mean(fit(probes x mean(transitions)))')
    except:
        pass

    # 1. fit of probes x transitions 2. mean of fits
    try:
        signif = SC_significance_dict[cell]
        dprime = SC_reals_dict[cell]
        signif_fits = list()
        dprime_fits = list()
        nprobes, ntrans, ntimes = SC_significance_dict[cell].shape
        for pp, tt in itt.product(range(nprobes), range(ntrans)):
            signif_fits.append(fit_line(times[:ntimes], signif[pp, tt, :]))
            dprime_fits.append(fit_line(times[:ntimes], dprime[pp, tt, :]))
        signif_fits = np.stack(signif_fits, axis=0)
        dprime_fits = np.stack(dprime_fits, axis=0)
        s_fit = np.mean(signif_fits, 0)
        d_fit = np.mean(dprime_fits, 0)
        axes[4].plot(times[:len(s_fit)], s_fit, color='black')
        twinx4 = axes[4].twinx()
        twinx4.plot(times[:len(d_fit)], d_fit, color='blue')
        axes[4].set_title('mean(fit(probe x transition))')
    except:
        pass

    # format the figure
    barax.get_shared_y_axes().join(barax, axes[1], axes[2], axes[3], axes[4])
    lineax.get_shared_y_axes().join(lineax, twinx1, twinx2, twinx3, twinx4)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, fig_folder, title)
    plt.close(fig)

# Second site comparisons, fit of the means or mean of the fits
site = sites[0]
for site in sites:
    fig_folder = 'site_dprime_signif_fits'
    title = f'{site}_dprime_signif_fits'
    fig_file = pl.Path(config['paths']['figures']) / f'{fig_folder}/{title}.png'
    # if fig_file.exists():
    #     print(f'{cell} figure alredy exists')
    #     continue
    # else:
    #     print(f'{cell} creating figure')

    full_signif = SC_significance_array[SC_sites_array == site]
    full_dprime = SC_reals_array[SC_sites_array == site]

    # plots cell mean significance and dprime
    fig, axes = plt.subplots(1, 3, figsize=full_screen)
    signif = np.mean(full_signif, (0, 1, 2))
    dprime = np.mean(full_dprime, (0, 1, 2))
    barkwargs = dict(width=bar_width, align='edge', color='black', edgecolor='white')
    linekwargs = dict(color='blue')
    barax, lineax = bar_line(times[:len(signif)], signif, dprime, ax=axes[0],
                             barkwargs=barkwargs, linekwargs=linekwargs)
    barax.set_title('full mean of values')

    # fit of the mean
    try:
        s_fit = fit_line(times[:len(signif)], signif)
        d_fit = fit_line(times[:len(dprime)], dprime)
        axes[1].plot(times[:len(s_fit)], s_fit, color='black')
        twinx1 = axes[1].twinx()
        twinx1.plot(times[:len(d_fit)], d_fit, color='blue')
        axes[1].set_title('fit of the mean')
    except:
        pass

    # mean of the fits
    try:
        signif_fits = list()
        dprime_fits = list()
        ncells, nprobes, ntrans, ntimes = full_signif.shape
        for cc, pp, tt in itt.product(range(ncells), range(nprobes), range(ntrans)):
            signif_fits.append(fit_line(times[:ntimes], full_signif[cc, pp, tt, :]))
            dprime_fits.append(fit_line(times[:ntimes], full_dprime[cc, pp, tt, :]))
        signif_fits = np.stack(signif_fits, axis=0)
        dprime_fits = np.stack(dprime_fits, axis=0)
        s_fit = np.mean(signif_fits, 0)
        d_fit = np.mean(dprime_fits, 0)
        axes[2].plot(times[:len(s_fit)], s_fit, color='black')
        twinx2 = axes[2].twinx()
        twinx2.plot(times[:len(d_fit)], d_fit, color='blue')
        axes[2].set_title('mean of the fits')
    except:
        pass
    # format the figure
    barax.get_shared_y_axes().join(barax, axes[1], axes[2])
    lineax.get_shared_y_axes().join(lineax, twinx1, twinx2)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, fig_folder, title)
    plt.close(fig)
