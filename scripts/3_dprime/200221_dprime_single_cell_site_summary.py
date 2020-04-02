import itertools as itt
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from progressbar import ProgressBar
from scipy.optimize import curve_fit

import cpn_dPCA as cdPCA
import cpn_dprime as cDP
import fancy_plots as fplt
from cpn_load import load, get_site_ids
from cpp_cache import make_cache, get_cache
from reliability import signal_reliability
from tools import shuffle_along_axis as shuffle

# todo edit motive
"""
Summary of the d' context discrimination significance, and propulation effect significance across all combinations of 
sites and probes.
The two metrics extracted are the total number of significant time bins and the position of the last time bin.

it is highly recomended to add a way of keeping track of the distibution of significant bins over time across each
category
"""


def exp(x, a, b):
    return a * np.exp(b * x)


def fit_exp_decay(times, values):
    """
    fits a properly constrained exponential decay to the times and values give, retursn the fitted values
    of the exponential function and the equivalent time constant Tau
    :param times: np.array. 1D, Time points in seconds, same shape as values
    :param values: np.array. 1D, y values, same shape as times
    :return:
    """
    if len(times) != len(values):
        times = times[:len(values)]

    #removes nan padding
    not_nan = np.logical_not(np.isnan(values))
    values = values[not_nan]
    times = times[not_nan]

    popt, pvar = curve_fit(exp, times, values, p0=[1, 0], bounds=([0, -np.inf], [np.inf, 0]))
    return popt, pvar


def plot_exp_decay(times, values, ax=None, label=True, pltkwargs={}):
    '''
    plots an exponential decaye curve fited on times and values
    :param times:
    :param values:
    :param ax:
    :param label:
    :param pltkwargs:
    :return:
    '''
    defaults = {'color': 'gray', 'linestyle': '--'}
    defaults.update(**pltkwargs)

    popt, pvar = fit_exp_decay(times, values)

    if ax == None:
        fig, ax = plt.subplots()
    else:
        ax = ax
        fig = ax.get_figure()

    if label == True:
        label = 'start={:+.2f}, tau= {:+.2f}'.format(popt[0], -1 / popt[1])
    elif label == False:
        label = None
    else:
        pass

    ax.plot(times, exp(times, *popt), color='gray', linestyle='--',
            label=label)

    return fig, ax, popt, pvar


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

    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, condition_axis=2)  # shape CellPair x Cell x Time

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

        shuffled.append(cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=3))

    shuffled = np.stack(shuffled, axis=1).squeeze(axis=0).swapaxes(0,1) # shape Montecarlo x ContextPair x Cell x Time

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
    dprime = cDP.pairwise_dprimes(dPCA_projection, observation_axis=0, condition_axis=1)

    # calculates floor (ctx shuffle) and ceiling (simulated data)
    sim_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))
    shuf_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))

    ctx_shuffle = trialR.copy()
    # pbar = ProgressBar()
    for rr in range(meta['montecarlo']):
        # ceiling: simulates data, calculates dprimes
        sim_trial = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0),
                                     size=[Re, C, S, T])
        sim_projection = cdPCA.transform(sim_trial, dPCA_transformation)
        sim_dprime[rr, ...] = cDP.pairwise_dprimes(sim_projection, observation_axis=0, condition_axis=1)

        ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)
        shuf_projection = cdPCA.transform(ctx_shuffle, dPCA_transformation)
        shuf_dprime[rr, ...] = cDP.pairwise_dprimes(shuf_projection, observation_axis=0, condition_axis=1)

    return dprime, shuf_dprime, sim_dprime, goodcells


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',  # blue, orange, green, brow,
                  '#984ea3', '#999999', '#e41a1c', '#dede00']  # purple, gray, scarlet, lime

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

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'montecarlo': 1000,
        'zscore': False}

tran_type = list()
for t0, t1 in itt.combinations(meta['transitions'], 2):
    pair = f'{t0}_{t1}'
    if pair == 'silence_continuous':
        tran_type.append('hard')
    elif pair == 'silence_similar':
        tran_type.append('hard')
    elif pair == 'silence_sharp':
        tran_type.append('soft')
    elif pair == 'continuous_similar':
        tran_type.append('soft')
    elif pair == 'continuous_sharp':
        tran_type.append('hard')
    elif pair == 'similar_sharp':
        tran_type.append('hard')
    else:
        continue

dprime_recache = False
rec_recache = False

all_probes = [2, 3, 5, 6]

sites = ['ley070a',  # good site. A1
         'ley072b',  # Primary looking responses with strong contextual effects
         'AMT028b',  # good site
         'AMT029a',  # Strong response, somehow visible contextual effects
         'AMT030a',  # low responses, Ok but not as good
         # 'AMT031a', # low response, bad
         'AMT032a']  # great site. PEG

sites = list(get_site_ids(316).keys())
# for site, probe in zip(['AMT029a', 'ley070a'],[5,2]):
# sites = ['AMT029a'] # example site PEG
# sites = ['ley070a'] # example site A1
# all_probes = [5]
# sites = ['DRX021a']

bad_SC_sites = list()
all_SC_pvalues = dict()
all_SC_reals = dict()
all_SC_shuffled = dict()

bad_dPCA_sites = list()
all_dPCA_pvalues = dict()
all_dPCA_reals = dict()
all_dPCA_shuffled = dict()
all_dPCA_simulated = dict()

for site in sites:

    this_site_SC_reals = list()
    this_site_SC_shuffled = list()
    this_site_SC_pvalues = list()

    this_site_dPCA_reals = list()
    this_site_dPCA_shuffled = list()
    this_site_dPCA_simulated = list()
    this_site_dPCA_pvalues = list()

    for pp, probe in enumerate(all_probes):
        ##############################
        # single cell analysis
        object_name = f'200221_{site}_P{probe}_single_cell_dprime'
        analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
        analysis_name = 'CPN_singel_cell_dprime'
        cache_folder = pl.Path('C:\\', 'users', 'mateo', 'mycache', analysis_name, analysis_parameters)

        SC_cache = make_cache(function=cell_dprime,
                              func_args={'site': site, 'probe': probe, 'meta': meta},
                              classobj_name=object_name,
                              cache_folder=cache_folder,
                              recache=dprime_recache)

        SC_dprime, SC_shuf_dprime, SC_cell_names, SC_trans_pairs = get_cache(SC_cache)

        this_site_SC_reals.append(SC_dprime)
        this_site_SC_shuffled.append(SC_shuf_dprime)

        # single tailed p value base on the montecarlo shuffling
        SC_pvalues = np.sum((SC_shuf_dprime >= SC_dprime), axis=0) / meta['montecarlo']
        this_site_SC_pvalues.append(SC_pvalues)

        ##############################
        # dPCA analysis
        object_name = f'200221_{site}_P{probe}_single_cell_dprime'
        analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
        analysis_name = 'CPN_dPCA_dprime'
        cache_folder = pl.Path('C:\\', 'users', 'mateo', 'mycache', analysis_name, analysis_parameters)

        dPCA_cache = make_cache(function=dPCA_fourway_analysis,
                                func_args={'site': site, 'probe': probe, 'meta': meta},
                                classobj_name=object_name,
                                cache_folder=cache_folder,
                                recache=dprime_recache)
        dPCA_dprime, dPCA_shuf_dprime, dPCA_sim_dprime, dPCA_cell_names = get_cache(dPCA_cache)

        this_site_dPCA_reals.append(dPCA_dprime)
        this_site_dPCA_shuffled.append(dPCA_shuf_dprime)
        this_site_dPCA_simulated.append(dPCA_sim_dprime)

        # single tailed p value base on the montecarlo shuffling
        dPCA_pvalues = np.sum((dPCA_shuf_dprime >= dPCA_dprime), axis=0) / meta['montecarlo']
        this_site_dPCA_pvalues.append(dPCA_pvalues)

    this_site_SC_reals = np.stack(this_site_SC_reals, axis=0)
    this_site_SC_shuffled = np.stack(this_site_SC_shuffled, axis=0)
    this_site_SC_pvalues = np.stack(this_site_SC_pvalues, axis=0)

    this_site_dPCA_reals = np.stack(this_site_dPCA_reals, axis=0)
    this_site_dPCA_shuffled = np.stack(this_site_dPCA_shuffled, axis=0)
    this_site_dPCA_simulated = np.stack(this_site_dPCA_simulated, axis=0)
    this_site_dPCA_pvalues = np.stack(this_site_dPCA_pvalues, axis=0)

    # reorders date in dictionary of cells
    for cc, cell in enumerate(SC_cell_names):
        all_SC_reals[cell] = this_site_SC_reals[:, :, cc, :]
        all_SC_shuffled[cell] = this_site_SC_shuffled[:, :, :, cc, :].swapaxes(0, 1)
        all_SC_pvalues[cell] = this_site_SC_pvalues[:, :, cc, :]

    all_dPCA_reals[site] = this_site_dPCA_reals
    all_dPCA_shuffled[site] = this_site_dPCA_shuffled.swapaxes(0, 1)
    all_dPCA_simulated[site] = this_site_dPCA_simulated.swapaxes(0, 1)
    all_dPCA_pvalues[site] = this_site_dPCA_pvalues

##############################
# defines arrays that identify cells, sites and regions
SC_cells = np.array(list(all_SC_pvalues.keys()))
SC_sites = np.array([cell[0:7] for cell in SC_cells])
SC_regions = np.array([cell[0:3] for cell in SC_cells]) # todo make a dictionary map from site to A1 or PEG

dPCA_site = np.array(list(all_dPCA_pvalues.keys()))
dPCA_regions = np.array([cell[0:3] for cell in dPCA_site])

# defines a significatn threshold and transfroms the pvalues into bool (significant vs nonsignificant)
threshold = 0.05
SC_significance = {key: (val <= threshold) for key, val in all_SC_pvalues.items()}
dPCA_significance = {key: (val <= threshold) for key, val in all_dPCA_pvalues.items()}


# stacks arrays, with different time dimentions, padding with NAN
SC_shape = np.insert(np.max(np.stack([arr.shape for arr in SC_significance.values()], axis=0), axis=0), 0, len(SC_significance))
SC_signif_array = np.empty(SC_shape)
SC_signif_array[:]=np.nan
for cc, arr in enumerate(SC_significance.values()):
    t = arr.shape[-1]
    SC_signif_array[cc, :, :, :t] = arr


dPCA_shape = np.insert(np.max(np.stack([arr.shape for arr in dPCA_significance.values()], axis=0), axis=0), 0, len(dPCA_significance))
dPCA_signif_array = np.empty(dPCA_shape)
dPCA_signif_array[:]=np.nan
for ss, arr in enumerate(dPCA_significance.values()):
    t = arr.shape[-1]
    dPCA_signif_array[ss, :, :, :t] = arr

# set up the time bin labels in milliseconds, this is critical fro ploting and calculating the tau
nbin = SC_signif_array.shape[-1]
fs = meta['raster_fs']
times = np.linspace(0, nbin / fs, nbin, endpoint=False) * 1000

############################################################
# dimensions to collapse per cell: Probe, Transition.
# collapse one, then the other, then both, per cell
# calculates the bins over/under a significant threshold
bar_width = 1 / fs * 1000

# for each cell in each site collapses across probes and context pairs
for site in sites:
    collapsed = np.nanmean(SC_signif_array[SC_sites == site, :, :, :], axis=(1, 2))
    site_cells = SC_cells[SC_sites == site]

    fig, axes = fplt.subplots_sqr(collapsed.shape[0], sp_kwargs={'sharey': True})
    for ax, hist, cell in zip(axes, collapsed, site_cells):
        ax.bar(times, hist, width=bar_width, align='edge', edgecolor='white')
        _ = plot_exp_decay(times, hist, ax=ax)
        ax.set_title(cell)
        ax.legend()

    fig.suptitle(site)

# for each site collapses across cells and probes
fig, axes = plt.subplots(len(sites),
                         len(list(itt.combinations(meta['transitions'], 2))),
                         sharex=True, sharey=True,squeeze=False)
for row, site in enumerate(sites):
    color = 'red' if site[0:3] == 'AMT' else 'black'
    collapsed = np.nanmean(SC_signif_array[SC_sites == site, :, :, :], axis=(0, 1))
    for col, (ax, hist, pair) in enumerate(
            zip(axes[row], collapsed, itt.combinations(meta['transitions'], 2))):
        ax.bar(times, hist, width=bar_width, align='edge', color=color, edgecolor='white')

        if row == 0:
            ax.set_title(f'{pair[0]}_{pair[1]}')

    if col == 0:
        ax.set_ylabel(site)

# for each site collapses across cells, probes, and transition pairs
fig, axes = plt.subplots(len(sites), 1,
                         sharex=True, sharey=True, squeeze=False)
for row, (site, ax) in enumerate(zip(sites, axes)):
    ax = ax[0]
    color = 'red' if site[0:3] == 'AMT' else 'black'
    collapsed = np.nanmean(SC_signif_array[SC_sites == site, :, :, :], axis=(0, 1, 2))

    max_sig = collapsed.max()
    ax.bar(times, collapsed, width=bar_width, align='edge', color=color, edgecolor='white')
    ax.set_ylabel(site)
    ax.legend()

# compares transitions: collapse across PEG vs A1, probe, cell
fig, axes = plt.subplots(2, 6, sharex=True, sharey=True)

for row, region in enumerate(['ley', 'AMT']):
    color = 'red' if region == 'AMT' else 'black'
    collapsed = np.nanmean(SC_signif_array[SC_regions == region, :, :, :], axis=(0, 1))

    for col, (ax, hist, pair) in enumerate(
            zip(axes[row], collapsed, itt.combinations(meta['transitions'], 2))):
        ax.bar(times, hist, width=bar_width, align='edge', color=color, edgecolor='white')

        if row == 0:
            ax.set_title(f'{pair[0]}_{pair[1]}')

        if col == 0:
            ax.set_ylabel(region)


############################################################
# for each site dPCA plots the significant bins for each transition pair and probe
for site, arr in dPCA_significance.items():
    fig, axes = plt.subplots(arr.shape[0], arr.shape[1], sharex=True, sharey=True, squeeze=False)
    color = 'black'
    for rr, (row, probe) in enumerate(zip(axes, all_probes)):
        for cc, (col, pair) in enumerate(zip(row, itt.combinations(meta['transitions'], 2))):
            ax = col
            hist = arr[rr,cc,:]
            ax.bar(times, hist, width=bar_width, align='edge', color=color, edgecolor='white')
            _ = plot_exp_decay(times, hist, ax=ax)

            if cc ==0:
                ax.set_xlabel('mean significant bins')
            if rr ==0:
                ax.set_title(f'{pair[0]}_{pair[1]}')

            ax.legend()
    fig.suptitle(f'{site}\n dPCA significant bins')

# for each site dPCA, plots the significant bins for each transitions pair, collapses probes
fig, axes = plt.subplots(len(dPCA_significance), arr.shape[1], sharex=True, sharey=True, squeeze=False)
for rr, (row, (site, arr)) in enumerate(zip(axes, dPCA_significance.items())):
    collapsed = np.nanmean(arr, axis=0)
    for cc, (col, pair) in enumerate(zip(row, itt.combinations(meta['transitions'], 2))):
        ax = col
        hist = collapsed[cc,:]
        ax.bar(times, hist, width=bar_width, align='edge', color=color, edgecolor='white')
        _ = plot_exp_decay(times, hist, ax=ax)

        if cc ==0:
            ax.set_xlabel(f'{site}\nmean significant bins')
        if rr ==0:
            ax.set_title(f'{pair[0]}_{pair[1]}')

        ax.legend()
    fig.suptitle(site)


# for each site collapses across cells, probes, and transition pairs, compares single cell with dPCA
fig, axes = plt.subplots(len(sites), 2,
                         sharex=True, sharey=True, squeeze=False)
for row, (site, ax) in enumerate(zip(sites, axes)):
    color = 'red' if site[0:3] == 'AMT' else 'black'
    SC_collapsed = np.nanmean(SC_signif_array[SC_sites == site, :, :, :], axis=(0, 1, 2))
    ax[0].bar(times, SC_collapsed, width=bar_width, align='edge', color=color, edgecolor='white')
    _ = plot_exp_decay(times, SC_collapsed, ax=ax[0])
    ax[0].set_ylabel(f'{site}\nmean significant bins')
    ax[0].set_title('mean of single cells')
    ax[0].legend()

    dPCA_collapsed = np.nanmean(dPCA_significance[site], axis=(0, 1))
    ax[1].bar(times[:len(dPCA_collapsed)], dPCA_collapsed, width=bar_width, align='edge', color=color, edgecolor='white')
    _ = plot_exp_decay(times, dPCA_collapsed, ax=ax[1])
    ax[1].set_title('dPCA')
    ax[1].legend()

fig.suptitle('single cell vs dPCA significant integration bins')

############################################################
# compares the Taus calculated for each cell against those of the site mean
# calculates exponential decay of significant bins for each cell, for the site mean and
# for the site dPCA projection collapsing across all probes and transisions

df = list()
for site in sites:
    collapsed = np.nanmean(SC_signif_array[SC_sites == site, :, :, :], axis=(1, 2))
    site_popt, _ = fit_exp_decay(times, np.nanmean(collapsed, axis=0))
    dPCA_popt, _ = fit_exp_decay(times, np.nanmean(dPCA_significance[site], axis=(0,1)))
    site_cells = SC_cells[SC_sites == site]
    for cell, hist in zip(site_cells, collapsed):
        cell_popt, _ = fit_exp_decay(times, hist)
        for sname, source in zip(('mean', 'dPCA', 'cell'),(site_popt, dPCA_popt, cell_popt)):
            for parameter, value in zip(('r0', 'decay'), source):
                d = {'siteid': site,
                     'cellid': cell,
                     'source': sname,
                     'parameter': parameter,
                     'value': value}
                df.append(d)
DF = pd.DataFrame(df)
# add tau defined as the inverse of the decay constant
decay = DF.loc[DF.parameter=='decay', :].copy()
decay['parameter'] = 'tau'
decay['value'] = -1 / decay['value']
DF = pd.concat((DF, decay), axis=0)

ff_param = DF.parameter == 'tau'
ff_threshold = DF.value < 1000 # drop cells with abnormaly high levels of tau, i.e, over 1 second
filtered = DF.loc[ff_param & ff_threshold, :]
pivoted = filtered.pivot(index='cellid', columns='source', values='value')

fig, ax = plt.subplots()
ax.scatter(pivoted['mean'], pivoted['dPCA'])
ax.plot(pivoted['cell'])

# plots all steps of analysis for this cell
def check_plot(cell, probe, tran_pair, significance=0.05):
    site = cell[:7]

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
    trialR = trialR.squeeze()

    cell_idx = goodcells.index(cell)
    prb_idx = all_probes.index(probe)
    t0_idx = meta['transitions'].index(tran_pair[0])
    t1_idx = meta['transitions'].index(tran_pair[1])
    pair_idx = SC_trans_pairs.index(f'{tran_pair[0]}_{tran_pair[1]}')

    fig, axes = plt.subplots(4, 1, sharex=True)
    axes = np.ravel(axes)

    # plots the PSTH and rasters of the compared responses
    axes[0].plot(times, trialR[:, cell_idx, t0_idx, :].mean(axis=0), color=trans_color_map[tran_pair[0]])
    axes[0].plot(times, trialR[:, cell_idx, t1_idx, :].mean(axis=0), color=trans_color_map[tran_pair[1]])

    bottom, top = axes[0].get_ylim()
    half = (top - bottom) / 2
    _ = fplt._raster(times, trialR[:, cell_idx, t0_idx, :], y_offset=0, y_range=(bottom, half), ax=axes[0],
                     scatter_kws={'color': trans_color_map[tran_pair[0]], 'alpha': 0.8, 's': 10})
    _ = fplt._raster(times, trialR[:, cell_idx, t1_idx, :], y_offset=0, y_range=(half, top), ax=axes[0],
                     scatter_kws={'color': trans_color_map[tran_pair[1]], 'alpha': 0.8, 's': 10})

    # plots the real dprime and the shuffled dprime
    axes[1].plot(times, all_SC_reals[cell][prb_idx, pair_idx, :], color='black')
    # axes[1].plot(all_shuffled[cell][:, prb_idx, pair_idx, :].T, color='green', alpha=0.01)
    _ = fplt._cint(times, all_SC_shuffled[cell][:, prb_idx, pair_idx, :], confidence=0.95, ax=axes[1],
                   fillkwargs={'color': 'black', 'alpha': 0.5})

    # plots the the calculated pvalue plust the significant threshold
    axes[2].plot(times, all_SC_pvalues[cell][prb_idx, pair_idx, :], color='black')
    axes[2].axhline(significance, color='red', linestyle='--')

    # plots the histogram of significant bins
    axes[3].bar(times, SC_significance[cell][prb_idx, pair_idx, :], width=bar_width, align='edge', edgecolor='white')
    return axes


cell = 'DRX021a-28-3'
axes = check_plot(cell, probe=6, tran_pair=('silence', 'continuous'), significance=threshold)
