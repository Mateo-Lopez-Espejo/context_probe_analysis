import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from scipy.io import loadmat
import fits as fts
import fancy_plots as fplt
from configparser import ConfigParser
import pathlib as pl
import joblib as jl
from cpp_cache import set_name
import pandas as pd
import itertools as itt
from fancy_plots import savefig


config = ConfigParser()
if pl.Path('../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../context_probe_analysis/config/settings.ini'))
elif pl.Path('../../../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../../../context_probe_analysis/config/settings.ini'))
else:
    raise FileNotFoundError('config file could not be find')

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

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None}

# load the calculated dprimes and montecarlo shuffling/simulations
# the loadede dictionary has 3 layers, analysis, value type and cell/site
batch_dprimes_file = pl.Path(config['path']['analysis_cache']) / 'batch_dprimes' / set_name(meta)
batch_dprimes = jl.load(batch_dprimes_file)
########################################################################################################################
# set up the time bin labels in milliseconds, this is critical for plotting and calculating the tau
nbin = np.max([value.shape[-1] for value in batch_dprimes['SC']['dprime'].values()])
fs = meta['raster_fs']
times = np.linspace(0, nbin / fs, nbin, endpoint=False) * 1000

bar_width = 1 / fs * 1000
fig_root = 'single_cell_context_dprime'


def compare_plot(cell):
    site = cell[0:7]

    SC_hist = np.nanmean(batch_dprimes['SC']['shuffled_significance'][cell], axis=(0, 1))
    SC_dprime = np.nanmean(batch_dprimes['SC']['dprime'][cell], axis=(0, 1))

    dPCA_hist = np.nanmean(batch_dprimes['dPCA']['shuffled_significance'][site], axis=(0, 1))
    dPCA_dprime = np.nanmean(batch_dprimes['dPCA']['dprime'][site], axis=(0, 1))

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
common_cells = all_cells.intersection(integration_cells)

# plots the comparison figures
cell = 'DRX021a-10-2'
cell = 'DRX008b-102-4'
for cell in common_cells:
    fig = compare_plot(cell)
    title = f'{cell} context vs integration'
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.suptitle(title)
    fig.set_size_inches([10.13, 9.74])
    fplt.savefig(fig, 'single_cell_comparison', title)
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
    signif = np.mean(batch_dprimes['SC']['shuffled_significance'][cell], (0, 1))
    dprime = np.mean(batch_dprimes['SC']['dprime'][cell], (0, 1))
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
        signif = np.mean(batch_dprimes['SC']['shuffled_significance'][cell], (0))
        dprime = np.mean(batch_dprimes['SC']['dprime'][cell], (0))
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
        signif = np.mean(batch_dprimes['SC']['shuffled_significance'][cell], (1))
        dprime = np.mean(batch_dprimes['SC']['dprime'][cell], (1))
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
        signif = batch_dprimes['SC']['shuffled_significance'][cell]
        dprime = batch_dprimes['SC']['dprime'][cell]
        signif_fits = list()
        dprime_fits = list()
        nprobes, ntrans, ntimes = batch_dprimes['SC']['shuffled_significance'][cell].shape
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

    full_signif = nanstack(
        [arr for key, arr in batch_dprimes['SC']['shuffled_significance'].items() if key[:7] == site])
    full_dprime = nanstack([arr for key, arr in batch_dprimes['SC']['dprime'].items() if key[:7] == site])

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
