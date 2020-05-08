import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl

import cpn_triplets as tp
from cpn_load import load
from reliability import signal_reliability

import cpn_dPCA as cdPCA

from scipy.stats import gaussian_kde as gkde
import seaborn as sn
import fancy_plots as plots



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

save_img = False


def weight_pdf(dpca, marginalization, axes=None, cellnames=None, color=None):

    if axes is None:
        fig, axes = plt.subplots(1)
        fig.suptitle('PDF marginalization weights')

    else:
        fig = axes.figure


    dd = dpca.P[marginalization][:,0]  # Neurons x Components
    pdf = gkde(dd)
    x = np.linspace(-1.2, 1.2, 100, endpoint=False)
    axes.plot(x, pdf(x), color=color, linewidth=2)
    axes = sn.swarmplot(x= dd, color=color, ax=axes)
    axes.set_ylim(-0.4, np.max(pdf(x)) + 0.2  )
    axes.set_yticks([0,1,2])
    axes.set_yticklabels([0, 1, 2])
    axes.set_xlabel('encoder weight')
    axes.set_ylabel('probability density')
    # axes.scatter(dd, np.zeros(len(dd)) + shuff, color=color, alpha=0.5)

    #
    # if cellnames is not None:
    #     cellnames = [cell[-4:] for cell in cellnames]
    #     ticks = dd.tolist()
    #     ticks.extend((-1, 1))
    #     print(ticks)
    #     tick_lables = cellnames.copy()
    #     tick_lables.extend(('-1', '1'))
    #     print(tick_lables)
    #     # todo diferenciate betwee minor and mayor ticks
    #     axes.set_xticks(ticks)
    #     axes.set_xticklabels(tick_lables, rotation='vertical')

    topcell = cellnames[np.argmax(dd)]

    return fig, axes, topcell




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
    for vv, ((marginalization, arr), color) in enumerate(zip(Z.items(), ['gray', 'green'])):

        ax = plt.subplot2grid((2,3), (vv, 1), 1, 1, fig=fig)


        for c in range(arr.shape[1]): # for each context

            toplot = arr[0, c, :] * meta['raster_fs'] # FixMe hardcoded firing rate
            ax.plot(time, toplot, label=meta['transitions'][c], color=CB_color_cycle[c], linewidth=2)
            ax.tick_params(labelsize='15')

        if meta['significance']:
            if marginalization in significance_masks:

                left, right = ax.get_xlim()
                bottom, top = ax.get_ylim()
                Ychunk = (top-bottom)/10
                ax.set_ylim(bottom-Ychunk, top)
                ax.imshow(significance_masks[marginalization][0][None, :],
                                    extent=[0, 1, bottom-Ychunk, bottom], aspect='auto', )

        ## Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # formats axes labels and ticks
        ax.set_ylabel(f'{code_to_name[marginalization]} dependent\nnormalized firing rate (Hz)',fontsize=20)

        if vv == len(Z)-1:
            ax.set_xlabel ('time (s)', fontsize=20)
            ax.legend()
        elif vv == 0:
            ax.set_title(f'1st principal component', fontsize=20)

        else:
            ax.axes.get_xaxis().set_visible(False)

        # marginalization weights
        weight_ax = plt.subplot2grid((2, 3), (vv, 2), 1, 1, fig=fig)
        fig, weight_ax, topcell = weight_pdf(dpca, marginalization=marginalization, axes=weight_ax,
                                    color=color, cellnames=sig.chans)
        print(topcell)
        weight_ax = weight_ax
        weight_ax.set_title('context linear weights')
        weight_ax.spines['right'].set_visible(False)
        weight_ax.spines['top'].set_visible(False)
        weight_ax.tick_params(labelsize=15)
        weight_ax.title.set_size(20)
        weight_ax.xaxis.label.set_size(20)
        weight_ax.yaxis.label.set_size(20)


    # plots variance explained
    var_ax = plt.subplot2grid((2,4), (0, 0), 1, 1, fig=fig )
    fig, var_ax, inset = cdPCA.variance_explained(dpca, ax=var_ax, names=['probe', 'context'], colors=['gray', 'green'])
    var_ax.set_title('variance explained')
    var_ax.spines['right'].set_visible(False)
    var_ax.spines['top'].set_visible(False)
    var_ax.tick_params(labelsize=15)
    var_ax.title.set_size(20)
    var_ax.xaxis.label.set_size(20)
    var_ax.yaxis.label.set_size(20)




    fig.suptitle(f'{site} probe {probe} dPCA', fontsize=20)

    # set figure to full size in tenrec screen
    fig.set_size_inches(19.2, 9.79)

    root = pl.Path(f'/home/mateo/Pictures/DAC2')
    if not root.exists(): root.mkdir(parents=True, exist_ok=True)
    png = root.joinpath(f'dPCA_{site}_P{probe}').with_suffix('.png')
    fig.savefig(png, transparent=True, dpi=100)
    svg = png = root.joinpath(f'dPCA_{site}_P{probe}').with_suffix('.svg')
    fig.savefig(svg, transparent=True)


    if site == 'AMT029a':
                        #  blue,   brown   orange     green
        CB_color_cycle = ['#377eb8','#a65628',  '#ff7f00', '#4daf4a', ]

    else:
                        #  blue,      orange     green      brown
        CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628']


    fig, ax = plots.hybrid(sig, f'\AC\d_P{probe}\Z', channels=topcell, time_strech=[1,2], colors=CB_color_cycle)
    ax = ax[0]
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=15)
    ax.title.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)

    fig.set_size_inches([4.83, 4.2 ])


    if save_img:
        root = pl.Path(f'/home/mateo/Pictures/DAC2')
        if not root.exists(): root.mkdir(parents=True, exist_ok=True)
        png = root.joinpath(f'raster_{site}_P{probe}').with_suffix('.png')
        fig.savefig(png, transparent=True, dpi=100)
        svg = png = root.joinpath(f'raster_{site}_P{probe}').with_suffix('.svg')
        fig.savefig(svg, transparent=True)

