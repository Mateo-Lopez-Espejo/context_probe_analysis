import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import joblib as jl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import src.visualization.fancy_plots
from src.data import LDA as cLDA, dPCA as cdPCA
from src.metrics import dprime as cDP
from src.visualization import fancy_plots as fplt
from src.data.load import load
from src.data.cache import set_name
from src.visualization.fancy_plots import savefig
from src.metrics.reliability import signal_reliability

"""
Summary of the d' context discrimination significance, and propulation effect significance across all combinations of 
sites and probes.
d'is calculated over single cell response or population dPCA
Multiple steps of the calculation process are displayed in figures for examples cells and sites 

"""

config = ConfigParser()
config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))

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
        'dprime_absolute': None}

rec_recache = False

region_map = dict(
    zip(['AMT028b', 'AMT029a', 'AMT030a', 'AMT031a', 'AMT032a', 'DRX008b', 'DRX021a', 'ley070a', 'ley072b'],
        ['PEG', 'PEG', 'PEG', 'PEG', 'PEG', 'A1', 'A1', 'A1', 'A1']))

all_probes = [2, 3, 5, 6]

# load the calculated dprimes and montecarlo shuffling/simulations
# the loadede dictionary has 3 layers, analysis, value type and cell/site
batch_dprimes_file = pl.Path(config['paths']['analysis_cache']) / 'batch_dprimes' / set_name(meta)
batch_dprimes = jl.load(batch_dprimes_file)

sites = set(batch_dprimes['dPCA']['dprime'].keys())
all_cells = set(batch_dprimes['SC']['dprime'].keys())

########################################################################################################################
# defines a significant threshold and transform the pvalues into boolean (significant vs nonsignificant)
threshold = 0.01
for analysis_name, mid_dict in batch_dprimes.items():
    mid_dict['shuffled_significance'] = {key: (val <= threshold) for key, val in mid_dict['shuffled_pvalue'].items()}
    if analysis_name != 'SC':
        mid_dict['simulated_significance'] = {key: (val <= threshold) for key, val in
                                             mid_dict['simulated_pvalue'].items()}

########################################################################################################################
# set up the time bin labels in milliseconds, this is critical for plotting and calculating the tau
nbin = np.max([value.shape[-1] for value in batch_dprimes['SC']['dprime'].values()])
fs = meta['raster_fs']
times = np.linspace(0, nbin / fs, nbin, endpoint=False) * 1000

bar_width = 1 / fs * 1000
fig_root = 'single_cell_context_dprime'


########################################################################################################################
########################################################################################################################
# plots all steps of analysis for example cell and site
def analysis_steps_plot(id, probe, source):
    site = id[:7] if source == 'SC' else id

    # loads the raw data
    recs = load(site, rasterfs=meta['raster_fs'], recache=False)
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

    if source == 'dPCA':
        projection, _ = cdPCA.fit_transform(R, trialR)
    elif source == 'LDA':
        projection, _ = cLDA.fit_transform_over_time(trialR)
        projection = projection.squeeze(axis=1)

    if meta['zscore'] is False:
        trialR = trialR * meta['raster_fs']
        if source == 'dPCA':
            projection = projection * meta['raster_fs']

    # flips signs of dprimes and montecarlos as needed
    dprimes, shuffleds = cDP.flip_dprimes(batch_dprimes[source]['dprime'][id],
                                          batch_dprimes[source]['shuffled_dprime'][id], flip='max')
    if source in ['dPCA', 'LDA']:
        _, simulations = cDP.flip_dprimes(batch_dprimes[source]['dprime'][id],
                                          batch_dprimes[source]['simulated_dprime'][id], flip='max')

    t = times[:trialR.shape[-1]]
    nrows = 2 if source == 'SC' else 3
    fig, axes = plt.subplots(nrows, 6, sharex='all', sharey='row')

    #  PSTH
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        t0_idx = meta['transitions'].index(trans[0])
        t1_idx = meta['transitions'].index(trans[1])

        if source == 'SC':
            cell_idx = goodcells.index(id)
            axes[0, tt].plot(t, trialR[:, cell_idx, t0_idx, :].mean(axis=0), color=trans_color_map[trans[0]],
                             linewidth=3)
            axes[0, tt].plot(t, trialR[:, cell_idx, t1_idx, :].mean(axis=0), color=trans_color_map[trans[1]],
                             linewidth=3)
        else:
            axes[0, tt].plot(t, projection[:, t0_idx, :].mean(axis=0), color=trans_color_map[trans[0]], linewidth=3)
            axes[0, tt].plot(t, projection[:, t1_idx, :].mean(axis=0), color=trans_color_map[trans[1]], linewidth=3)

    # Raster, dprime, CI
    bottom, top = axes[0, 0].get_ylim()
    half = ((top - bottom) / 2) + bottom
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        prb_idx = all_probes.index(probe)
        pair_idx = tt

        if source == 'SC':
            # raster
            cell_idx = goodcells.index(id)
            t0_idx = meta['transitions'].index(trans[0])
            t1_idx = meta['transitions'].index(trans[1])

            _ = fplt._raster(t, trialR[:, cell_idx, t0_idx, :], y_offset=0, y_range=(bottom, half), ax=axes[0, tt],
                             scatter_kws={'color': trans_color_map[trans[0]], 'alpha': 0.4, 's': 10})
            _ = fplt._raster(t, trialR[:, cell_idx, t1_idx, :], y_offset=0, y_range=(half, top), ax=axes[0, tt],
                             scatter_kws={'color': trans_color_map[trans[1]], 'alpha': 0.4, 's': 10})

        # plots the real dprime and the shuffled dprime ci
        axes[1, tt].plot(t, dprimes[prb_idx, pair_idx, :], color='black')
        _ = fplt._cint(t, shuffleds[:, prb_idx, pair_idx, :], confidence=0.95, ax=axes[1, tt],
                       fillkwargs={'color': 'black', 'alpha': 0.5})

        if source in ['dPCA', 'LDA']:
            # plots the real dprime and simulated dprime ci
            axes[2, tt].plot(t, dprimes[prb_idx, pair_idx, :], color='black')
            _ = fplt._cint(t, simulations[:, prb_idx, pair_idx, :], confidence=0.95, ax=axes[2, tt],
                           fillkwargs={'color': 'black', 'alpha': 0.5})

    # significance bars
    ax1_bottom = axes[1, 0].get_ylim()[0]
    if source == 'dPCA':
        ax2_bottom = axes[2, 0].get_ylim()[0]
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        prb_idx = all_probes.index(probe)
        pair_idx = tt
        # histogram of context discrimination
        axes[1, tt].bar(t, batch_dprimes[source]['shuffled_significance'][id][prb_idx, pair_idx, :],
                        width=bar_width, align='center', edgecolor='white', bottom=ax1_bottom)
        if source in ['dPCA', 'LDA']:
            # histogram of population effects
            axes[2, tt].bar(t, batch_dprimes[source]['simulated_significance'][id][prb_idx, pair_idx, :],
                            width=bar_width, align='center', edgecolor='white', bottom=ax2_bottom)

        # formats legend
        if tt == 0:
            axes[0, tt].set_ylabel(f'dPC', fontsize=ax_lab_size)
            axes[0, tt].tick_params(labelsize=ax_val_size)
            axes[1, tt].set_ylabel(f'dprime', fontsize=ax_lab_size)
            axes[1, tt].tick_params(labelsize=ax_val_size)
            if source in ['dPCA', 'LDA']:
                axes[2, tt].set_ylabel(f'dprime', fontsize=ax_lab_size)
                axes[2, tt].tick_params(labelsize=ax_val_size)

        axes[-1, tt].set_xlabel('time (ms)', fontsize=ax_lab_size)
        axes[-1, tt].tick_params(labelsize=ax_val_size)
        axes[0, tt].set_title(f'{trans[0]}_{trans[1]}', fontsize=sub_title_size)

        for ax in np.ravel(axes):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    return fig, axes


for cell in ['AMT028b-20-1', 'DRX008b-04-1']:
    probe = 2
    fig, axes = analysis_steps_plot(cell, probe, 'SC')
    half_screen = (full_screen[0], full_screen[1] / 2)
    fig.set_size_inches(half_screen)
    title = f'SC, {cell} probe {probe} calc steps'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, 'DAC3_figures', title)

for site in ['AMT028b', 'DRX008b']:
    probe = 2
    fig, axes = analysis_steps_plot(site, probe, 'dPCA')
    half_screen = (full_screen[0], full_screen[1] / 2)
    fig.set_size_inches(half_screen)
    title = f'dPCA, {site} probe {probe}, calc steps'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, 'DAC3_figures', title)

for site in ['AMT028b', 'DRX008b']:
    probe = 2
    fig, axes = analysis_steps_plot(site, probe, 'LDA')
    half_screen = (full_screen[0], full_screen[1] / 2)
    fig.set_size_inches(half_screen)
    title = f'LDA, {site} probe {probe}, calc steps'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fplt.savefig(fig, 'DAC3_figures', title)


########################################################################################################################
# plots the summary for all categories (probe, context pairs)
def category_summary_plot(id, source):
    """
    Plots calculated dprime, confidense interval of shuffled dprime, and histogram of significant bins, for all contexts
    and probes.
    Subplots are a grid of al combinations of probe (rows) and context pairs (columns), plus the means of each category,
    and the grand mean
    :param id: str. cell or site id
    :param source: str. 'SC', 'dPCA', or 'LDA'
    :return: fig, axes.
    """

    # flips signs of dprimes and montecarlos as neede
    dprimes, shuffleds = cDP.flip_dprimes(batch_dprimes[source]['dprime'][id],
                                          batch_dprimes[source]['shuffled_dprime'][id], flip='max')
    signif_bars = batch_dprimes[source]['shuffled_significance'][id]

    t = times[:dprimes.shape[-1]]
    fig, axes = plt.subplots(5, 7, sharex='all', sharey='all')

    # dprime and confidence interval for each probe-transition combinations
    for (pp, probe), (tt, trans) in itt.product(enumerate(all_probes),
                                                enumerate(itt.combinations(meta['transitions'], 2))):
        prb_idx = all_probes.index(probe)

        # plots the real dprime and the shuffled dprime
        axes[pp, tt].plot(t, dprimes[prb_idx, tt, :], color='black')
        _ = fplt._cint(t, shuffleds[:, prb_idx, tt, :], confidence=0.95, ax=axes[pp, tt],
                       fillkwargs={'color': 'black', 'alpha': 0.5})
    # dprime and ci for the mean across context pairs
    for pp, probe in enumerate(all_probes):
        prb_idx = all_probes.index(probe)
        axes[pp, -1].plot(t, np.mean(dprimes[prb_idx, :, :], axis=0), color='black')
        axes[pp, -1].axhline(0, color='gray', linestyle='--')

    # dprime and ci for the mean across probes
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        axes[-1, tt].plot(t, np.mean(dprimes[:, tt, :], axis=0), color='black')
        axes[-1, tt].axhline(0, color='gray', linestyle='--')

    # significance bars for each probe-transition combinations
    bar_bottom = axes[0, 0].get_ylim()[0]
    for (pp, probe), (tt, trans) in itt.product(enumerate(all_probes),
                                                enumerate(itt.combinations(meta['transitions'], 2))):
        prb_idx = all_probes.index(probe)
        axes[pp, tt].bar(t, signif_bars[prb_idx, tt, :], width=bar_width, align='center',
                         edgecolor='white', bottom=bar_bottom)
        # _ = fplt.exp_decay(t, signif_bars[prb_idx, tt, :], ax=axes[2, tt])

    # significance bars for the mean across context pairs
    for pp, probe in enumerate(all_probes):
        prb_idx = all_probes.index(probe)
        axes[pp, -1].bar(t, np.mean(signif_bars[prb_idx, :, :], axis=0), width=bar_width, align='center',
                         edgecolor='white', bottom=bar_bottom)
        _ = fplt.exp_decay(t, np.mean(signif_bars[prb_idx, :, :], axis=0), ax=axes[pp, -1], yoffset=bar_bottom,
                           linestyle='.', color='gray')
        axes[pp, -1].legend(loc='upper right', fontsize='small', markerscale=3, frameon=False)

    # significance bars for the mean across probes
    for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
        axes[-1, tt].bar(t, np.mean(signif_bars[:, tt, :], axis=0), width=bar_width, align='center',
                         edgecolor='white', bottom=bar_bottom)
        _ = fplt.exp_decay(t, np.mean(signif_bars[:, tt, :], axis=0), axes[-1, tt], yoffset=bar_bottom,
                           linestyle='.', color='gray')
        axes[-1, tt].legend(loc='upper right', fontsize='small', markerscale=3, frameon=False)

    # cell summary mean: dprime, confidence interval
    axes[-1, -1].plot(t, np.mean(dprimes[:, :, :], axis=(0, 1)), color='black')
    axes[-1, -1].axhline(0, color='gray', linestyle='--')
    axes[-1, -1].bar(t, np.mean(signif_bars[:, :, :], axis=(0, 1)), width=bar_width, align='center',
                     edgecolor='white', bottom=bar_bottom)
    _ = fplt.exp_decay(t, np.mean(signif_bars[:, :, :], axis=(0, 1)), ax=axes[-1, -1], yoffset=bar_bottom,
                       linestyle='.', color='gray')
    axes[-1, -1].legend(loc='upper right', fontsize='small', markerscale=3, frameon=False)

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


# SC example
for cell in ['AMT028b-20-1', 'DRX008b-04-1']:
    fig, axes = category_summary_plot(cell, 'SC')
    fig.set_size_inches(full_screen)
    title = f'SC, {cell} probe pair summary'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, 'DAC3_figures', title)

# dpca site example
for site in ['AMT028b', 'DRX008b']:
    fig, axes = category_summary_plot(site, 'dPCA')
    fig.set_size_inches(full_screen)
    title = f'dPCA, {site} probe pair summary'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, 'DAC3_figures', title)

for site in ['AMT028b', 'DRX008b']:
    fig, axes = category_summary_plot(site, 'lDA')
    fig.set_size_inches(full_screen)
    title = f'LDA, {site} probe pair summary'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fplt.savefig(fig, 'DAC3_figures', title)


########################################################################################################################
# fit and metrics example, taking the man across probe and context pairs
def fit_example_plot(id, source):
    """
    Plots dprime (top) and significant bins (bottom) with their fitted exponential decays. Both dprime and significant
    bins are the grand mean across all probes and context pairs for a given cell or site.
    :param id: str. cell or site id
    :param source: str. 'SC', 'dPCA', or 'LDA'
    :return: fig, axes.
    """

    # flips signs of dprimes and montecarlos as neede
    dprimes, shuffleds = cDP.flip_dprimes(batch_dprimes[source]['dprime'][id],
                                          batch_dprimes[source]['shuffled_dprime'][id], flip='max')
    signif_bars = batch_dprimes[source]['shuffled_significance'][id]

    mean_dprime = np.mean(dprimes[:, :, :], axis=(0, 1))
    mean_signif = np.mean(signif_bars[:, :, :], axis=(0, 1))

    t = times[:dprimes.shape[-1]]
    fig, axes = plt.subplots(2, 1, sharex='all', sharey='all')

    # plots dprime plus fit
    axes[0].plot(t, mean_dprime, color='black')
    axes[0].axhline(0, color='gray', linestyle='--')
    _ = fplt.exp_decay(t, mean_dprime, ax=axes[0], linestyle='--', color='black')

    # plots confifence bins plut fit
    axes[1].bar(t, mean_signif, width=bar_width, align='center',
                edgecolor='white', )
    _ = fplt.exp_decay(times, mean_signif, ax=axes[1], linestyle='--', color='black')

    axes[0].legend(loc='upper right', fontsize=ax_val_size, markerscale=3, frameon=False)
    axes[1].legend(loc='upper right', fontsize=ax_val_size, markerscale=3, frameon=False)

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


# SC examples
for cell in ['AMT028b-20-1', 'DRX008b-04-1']:
    fig, axes = fit_example_plot(cell, 'SC')
    fig.set_size_inches((8, 8))
    title = f'SC, {cell} fit summary'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, 'DAC3_figures', title)

# dPCA site examples
for site in ['AMT028b', 'DRX008b']:
    fig, axes = fit_example_plot(site, 'dPCA')
    fig.set_size_inches((8, 8))
    title = f'dPCA, {site} fit summary'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, 'DAC3_figures', title)


########################################################################################################################
# dprimes, hist and fit of all cells in a site, grand mean across transition pairs and probes
def site_cell_summary(id):
    """
    plots a grid of subplots, each one showing the real dprime, histogram of significant bins and fitted exponential
    decay to the significant bins. Both the dprime and significant bins are the cell grand mean across probes and
    context pairs
    :param id: str. site id
    :return: fig, axes
    """

    site_cells = set([cell for cell in batch_dprimes['SC']['dprime'].keys() if cell[:7] == id])

    fig, axes = fplt.subplots_sqr(len(site_cells), sharex=True, sharey=True)
    for ax, cell in zip(axes, site_cells):
        grand_mean, _ = cDP.flip_dprimes(batch_dprimes['SC']['dprime'][cell], flip='max')
        line = np.mean(grand_mean, axis=(0, 1))
        hist = np.mean(batch_dprimes['SC']['shuffled_significance'][cell], axis=(0, 1))
        ax.plot(times[:len(line)], line, color='black')
        ax.bar(times[:len(hist)], hist, width=bar_width, align='center', color='C0', edgecolor='white')
        _ = fplt.exp_decay(times[:len(hist)], hist, ax=ax, linestyle='--', color='gray')
        # ax.set_title(cell, fontsize=10)
        ax.legend(loc='upper right', fontsize='small', markerscale=3, frameon=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    return fig, axes


for site in ['AMT028b', 'DRX008b']:
    fig, axes = site_cell_summary(site)
    fig.set_size_inches(full_screen)
    title = f'{site} all cells summary'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig, 'wip3_figures', title)


########################################################################################################################
# dPCA site summary showing dPCs and explained variance

def dPCA_site_summary(site, probe):
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
    Z, trialZ, dpca = cdPCA.trials_dpca(R, trialR)

    fig, axes = plt.subplots(2, 3, sharex='all', sharey='row')
    for vv, (marginalization, arr) in enumerate(Z.items()):
        means = Z[marginalization]
        trials = trialZ[marginalization]

        if marginalization == 't':
            marginalization = 'probe'
        elif marginalization == 'ct':
            marginalization = 'context'
        for pc in range(3):  # first 3 principal components

            ax = axes[vv, pc]
            for tt, trans in enumerate(meta['transitions']):  # for each context
                ax.plot(times, means[pc, tt, :], label=trans, color=trans_color_map[trans], linewidth=2)
                # _ = fplt._cint(times, trials[:,pc,tt,:],  confidence=0.95, ax=ax,
                #                fillkwargs={'color': trans_color_map[trans], 'alpha': 0.5})
                ax.tick_params(labelsize=ax_val_size)

            # formats axes labels and ticks
            if pc == 0:  # y labels
                ax.set_ylabel(f'{marginalization} dependent\nfiring rate (z-score)', fontsize=ax_lab_size)
            else:
                ax.axes.get_yaxis().set_visible(True)
                pass

            if vv == 0:
                ax.set_title(f'dPC #{pc + 1}', fontsize=sub_title_size)
                ax.axes.get_xaxis().set_visible(True)
            else:
                ax.set_xlabel('time (ms)', fontsize=ax_lab_size)

            ## Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(labelsize=ax_val_size)
    # legend in last axis
    axes[-1, -1].legend(loc='upper right', fontsize='x-large', markerscale=10, frameon=False)

    return fig, ax, dpca


def var_explained(dpca):
    # plots variance explained
    fig, var_ax = plt.subplots()
    fig, var_ax, inset = src.visualization.fancy_plots.variance_explained(dpca, ax=var_ax, names=['probe', 'context'], colors=['gray', 'green'])
    _, labels, autotexts = inset
    plt.setp(autotexts, size=15, weight='normal')
    plt.setp(labels, size=15, weight='normal')
    var_ax.set_title('marginalized variance')
    var_ax.spines['right'].set_visible(False)
    var_ax.spines['top'].set_visible(False)
    var_ax.tick_params(labelsize=ax_val_size)
    var_ax.title.set_size(sub_title_size)
    var_ax.xaxis.label.set_size(ax_lab_size)
    var_ax.yaxis.label.set_size(ax_lab_size)
    return fig, var_ax


for site in ['AMT028b', 'DRX008b']:
    probe = 2
    fig1, axes, dpca = dPCA_site_summary(site, probe)
    fig1.set_size_inches((12, 8))
    title = f'{site} probe-{probe} dPCA projection'
    fig1.suptitle(title)
    fig1.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig1, 'wip3_figures', title)

    fig2, ax = var_explained(dpca)
    fig2.set_size_inches((6, 6))
    title = f'{site} probe-{probe} dPCA variance explained'
    fig2.suptitle(title)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    savefig(fig2, 'wip3_figures', title)
