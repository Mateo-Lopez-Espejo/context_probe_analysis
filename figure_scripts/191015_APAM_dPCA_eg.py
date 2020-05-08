import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl

import cpn_triplets as tp
from cpn_load import load
from reliability import signal_reliability

import cpn_dPCA as cdPCA
import fancy_plots as cplt

"""
plots an example dPCA analysis with variance explained, context and probe marginalization projections
and hybrid raster/PSTH plot for the most weigted cell in the context marginalization

does this for two example site/probes
"""

plt.rcParams['svg.fonttype'] = 'none'

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',  # blue, orange, green, brow,
                  '#984ea3', '#999999', '#e41a1c', '#dede00']  # purple, gray, scarlet, lime

trans_color_map = {'silence': '#377eb8',  # blue
                   'continuous': '#ff7f00',  # orange
                   'similar': '#4daf4a',  # green
                   'sharp': '#a65628'}  # brown

# meta parameter
meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot': [2, 3, 5, 6],
        'significance': False,
        'zscore': False}

transitions = {'P2': {'silence': 0,
                      'continuous': 1,
                      'similar': 3,
                      'sharp': 6},
               'P3': {'silence': 0,
                      'continuous': 2,
                      'similar': 1,
                      'sharp': 5},
               'P5': {'silence': 0,
                      'continuous': 4,
                      'similar': 6,
                      'sharp': 3},
               'P6': {'silence': 0,
                      'continuous': 5,
                      'similar': 4,
                      'sharp': 2}}

code_to_name = {'t': 'Probe', 'ct': 'Context'}

for site, probe in zip(['AMT029a', 'ley070a'], [5, 2]):
    # load and format triplets from a site
    # site = 'AMT030a' # low responses, Ok but not as good
    recs = load(site)
    rec = recs['trip0']
    sig = rec['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig.rasterize(), r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # get a specific probe after a set of different transitions
    Z, trialZ, significance_masks, dpca = cdPCA.tran_dpca(sig, probe, channels=goodcells,
                                                          transitions=meta['transitions'],
                                                          smooth_window=meta['smoothing_window'],
                                                          significance=meta['significance'],
                                                          raster_fs=meta['raster_fs'],
                                                          part='probe',
                                                          zscore=meta['zscore'])

    expl_var = dpca.explained_variance_ratio_

    # plots the first PC projection of each context, for each marginalization

    fig = plt.figure()
    T = trialZ['ct'].shape[-1]
    time = np.linspace(0, T / meta['raster_fs'], T, endpoint=False)
    for vv, (marginalization, arr) in enumerate(Z.items()):

        for pc in range(1):  # first 3 principal components

            PC_ax = plt.subplot2grid((2, 2), (vv, pc + 1), 1, 1, fig=fig)

            for c in range(arr.shape[1]):  # for each context

                toplot = arr[pc, c, :] * meta['raster_fs']  # FixMe hardcoded firing rate
                PC_ax.plot(time, toplot, label=meta['transitions'][c], color=CB_color_cycle[c], linewidth=2)
                PC_ax.tick_params(labelsize='15')

            if meta['significance']:
                if marginalization in significance_masks:
                    left, right = PC_ax.get_xlim()
                    bottom, top = PC_ax.get_ylim()
                    Ychunk = (top - bottom) / 10
                    PC_ax.set_ylim(bottom - Ychunk, top)
                    PC_ax.imshow(significance_masks[marginalization][pc][None, :],
                                 extent=[0, 1, bottom - Ychunk, bottom], aspect='auto', )

            ## Hide the right and top spines
            PC_ax.spines['right'].set_visible(False)
            PC_ax.spines['top'].set_visible(False)

            # formats axes labels and ticks
            if pc == 0:  # y labels
                PC_ax.set_ylabel(f'{code_to_name[marginalization]} dependent\nnormalized firing rate (Hz)', fontsize=20)
            else:
                PC_ax.axes.get_yaxis().set_visible(False)

            if vv == len(Z) - 1:
                PC_ax.set_xlabel('time (s)', fontsize=20)
                if pc == 2:  # bottom right coner subplt:
                    PC_ax.legend()
            elif vv == 0:
                PC_ax.set_title(f'{pc + 1}th principal component', fontsize=20)

            else:
                PC_ax.axes.get_xaxis().set_visible(False)

    # plots variance explained
    var_ax = plt.subplot2grid((2, 4), (0, 0), 1, 1, fig=fig)
    cdPCA.variance_explained(dpca, ax=var_ax, names=['probe', 'context'], colors=['gray', 'green'])
    var_ax.set_title('variance explained')
    var_ax.spines['right'].set_visible(False)
    var_ax.spines['top'].set_visible(False)
    var_ax.tick_params(labelsize=15)
    var_ax.title.set_size(20)
    var_ax.xaxis.label.set_size(20)
    var_ax.yaxis.label.set_size(20)

    # plots example raster
    epoch_names = [f"C{transitions[f'P{probe}'][trans]}_P{probe}" for trans in meta['transitions']]
    topcell = goodcells[np.argmax(np.abs(dpca.D['ct'][:, 0]))]
    colors = [trans_color_map[trans] for trans in meta['transitions']]

    raster_ax = plt.subplot2grid((2, 2), (1, 0), 1, 1, fig=fig)
    cplt.hybrid(sig, epoch_names=epoch_names, channels=topcell, time_strech=[1, 2], colors=colors, axes=[raster_ax])
    raster_ax = raster_ax
    raster_ax.spines['right'].set_visible(False)
    raster_ax.spines['top'].set_visible(False)
    raster_ax.tick_params(labelsize=15)
    raster_ax.title.set_size(20)
    raster_ax.xaxis.label.set_size(20)
    raster_ax.yaxis.label.set_size(20)

    suptitle = f"{site} probe {probe} dPCA zscore-{meta['zscore']}"
    fig.suptitle(suptitle, fontsize=20)

    analysis = f"dPCA_examples_{meta['raster_fs']}Hz_zscore-{meta['zscore']}"

    # set figure to full size in tenrec screen
    fig.set_size_inches(9, 7)

    # root = pl.Path(f'/home/mateo/Pictures/APAM/{analysis}')
    # if not root.exists(): root.mkdir(parents=True, exist_ok=True)
    # png = root.joinpath(suptitle).with_suffix('.png')
    # fig.savefig(png, transparent=True, dpi=100)
    # svg = png = root.joinpath(suptitle).with_suffix('.svg')
    # fig.savefig(svg, transparent=True)
