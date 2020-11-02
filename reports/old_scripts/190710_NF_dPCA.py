import matplotlib.pyplot as plt
import numpy as np

import src.visualization.fancy_plots
from src.data.load import load
from src.metrics.reliability import signal_reliability

from src.data import dPCA as cdPCA, triplets as tp

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',
                  '#984ea3', '#999999', '#e41a1c', '#dede00']

# meta parameter
meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 20, # ms
        'raster_fs': 100,
        'transitions' : ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot' : [2,3,5,6],
        'significance': False}



code_to_name = {'t': 'Probe', 'ct': 'Context'}

for site, probe in zip(['AMT029a', 'ley070a'],[5,2]):
    # load and format triplets from a site
    # site = 'AMT030a' # low responses, Ok but not as good
    recs = load(site)
    rec = recs['trip0']
    sig = rec['resp'].rasterize()

    # calculates response realiability and select only good cells to improve analysis

    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    if len(goodcells) < 10:
        n_components = len(goodcells)
    elif len(goodcells) == 0:
        pass#continue
    else:
        n_components = 10
    # plots PSTHs of all probes after silence
    # fig, axes = cplot.hybrid(sig, epoch_names=r'\AC0_P[2356]\Z', channels=goodcells)

    # plots PSHTs of individual best probe after all contexts
    # fig, axes = cplot.hybrid(sig, epoch_names=r'\AC\d_P3\Z', channels=goodcells)

    # takes an example probe
    full_array, invalid_cp, valid_cp, all_contexts, all_probes = \
        tp.make_full_array(sig, channels=goodcells, smooth_window=meta['smoothing_window'])



    # get a specific probe after a set of different transitions

    Z, trialZ, significance_masks, dpca = cdPCA.tran_dpca(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                                 smooth_window=meta['smoothing_window'], significance=meta['significance'],
                                                 raster_fs=meta['raster_fs'])

    expl_var = dpca.explained_variance_ratio_

    # plots the first PC projection of each context, for each marginalization
    # includes a measurement of significance by shuffle test

    fig = plt.figure()

    time = np.linspace(0,1,100, endpoint=False)
    for vv, (marginalization, arr) in enumerate(Z.items()):

        for pc in range(3): # first 3 principal components

            if pc == 0:
                ax = plt.subplot2grid((2,4), (vv, pc+1), 1, 1, fig=fig )
                refax = ax
            else:
                ax = plt.subplot2grid((2,4), (vv, pc+1), 1, 1, fig=fig, sharey=refax)


            for c in range(arr.shape[1]): # for each context

                toplot = arr[pc, c, :] * meta['raster_fs'] # FixMe hardcoded firing rate
                ax.plot(time, toplot, label=meta['transitions'][c], color=CB_color_cycle[c], linewidth=2)
                ax.tick_params(labelsize='15')

            if meta['significance']:
                if marginalization in significance_masks:

                    left, right = ax.get_xlim()
                    bottom, top = ax.get_ylim()
                    Ychunk = (top-bottom)/10
                    ax.set_ylim(bottom-Ychunk, top)
                    ax.imshow(significance_masks[marginalization][pc][None, :],
                                        extent=[0, 1, bottom-Ychunk, bottom], aspect='auto', )

            ## Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # formats axes labels and ticks
            if pc == 0:  # y labels
                ax.set_ylabel(f'{code_to_name[marginalization]} dependent\nnormalized firing rate (Hz)',fontsize=20)
            else:
                ax.axes.get_yaxis().set_visible(False)

            if vv == len(Z)-1:
                ax.set_xlabel ('time (s)', fontsize=20)
                if pc == 2: # bottom right coner subplt:
                    ax.legend()
            elif vv == 0:
                ax.set_title(f'{pc}th principal component', fontsize=20)

            else:
                ax.axes.get_xaxis().set_visible(False)




    # plots variance explained
    var_ax = plt.subplot2grid((2,4), (0, 0), 1, 1, fig=fig )
    fig, var_ax, inset = src.visualization.fancy_plots.variance_explained(dpca, ax=var_ax, names=['probe', 'context'], colors=['gray', 'green'])
    var_ax.set_title('variance explained')
    var_ax.spines['right'].set_visible(False)
    var_ax.spines['top'].set_visible(False)
    var_ax.tick_params(labelsize=15)
    var_ax.title.set_size(20)
    var_ax.xaxis.label.set_size(20)
    var_ax.yaxis.label.set_size(20)

    # marginalization weights
    weight_ax = plt.subplot2grid((2,4), (1, 0), 1, 1, fig=fig)
    fig, weight_ax = src.visualization.fancy_plots.weight_pdf(dpca, marginalization=['ct'], axes=weight_ax,
                                                              only_first=False, color=['gray', 'green'])
    weight_ax = weight_ax[0]
    weight_ax.set_title('context linear weights')
    weight_ax.spines['right'].set_visible(False)
    weight_ax.spines['top'].set_visible(False)
    weight_ax.tick_params(labelsize=15)
    weight_ax.title.set_size(20)
    weight_ax.xaxis.label.set_size(20)
    weight_ax.yaxis.label.set_size(20)


    fig.suptitle(f'{site} probe {probe} dPCA', fontsize=20)

    # set figure to full size in tenrec screen
    fig.set_size_inches(19.2, 9.79)

    # root = pl.Path(f'/home/mateo/Pictures/DAC2')
    # if not root.exists(): root.mkdir(parents=True, exist_ok=True)
    # png = root.joinpath(f'dPCA_{site}_P{probe}').with_suffix('.png')
    # fig.savefig(png, transparent=True, dpi=100)
    # svg = png = root.joinpath(f'dPCA_{site}_P{probe}').with_suffix('.svg')
    # fig.savefig(svg, transparent=True)

