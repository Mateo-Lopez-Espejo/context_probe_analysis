import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

import cpn_dPCA as cdPCA
import cpn_LDA as cLDA

import cpn_dprime as cDP
import fancy_plots
import fancy_plots as fplt
from cpn_batch_dprime import LDA_fourway_analysis
from cpn_load import load
from cpp_cache import make_cache, get_cache
from reliability import signal_reliability
from tools import shuffle_along_axis as shuffle

"""
Summary of the d' context discrimination significance, and propulation effect significance across all combinations of 
sites and probes. In particular comparing the significance of population dprimes from LDA or dPCA dimensionality reductions

This tries to recapitulate some old comparison, but this time using the more streamlined dprime analysis

overall it seems that LDA is to efficient at capturing arbitrary discrimination hiperplanes, thus it significantly 
increases the dprimes of shuffled and simulated trials, increasing the nonsignificant floor and thus decreasing the
overall significance of the real dprime
"""

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
    fplt.savefig(fig, 'DAC3_figures', title)

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
    fplt.savefig(fig, 'DAC3_figures', title)