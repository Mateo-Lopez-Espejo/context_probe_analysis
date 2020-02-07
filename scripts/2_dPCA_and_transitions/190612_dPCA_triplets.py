import os

import matplotlib.pyplot as plt
import numpy as np
from dPCA import dPCA

import fancy_plots as cplot

import cpn_triplets as tp
from cpn_load import load
from reliability import signal_reliability
from cpp_parameter_handlers import _channel_handler
import cpn_dPCA as cdPCA

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

all_sites = ['ley070a', # good site. A1
             'ley072b', # Primary looking responses with strong contextual effects
             'AMT028b', # good site
             'AMT029a', # Strong response, somehow visible contextual effects
             'AMT030a', # low responses, Ok but not as good
             #'AMT031a', # low response, bad
             'AMT032a'] # great site. PEG


# meta parameter
meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 20, # ms
        'transitions' : ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot' : [2,3,5,6],
        'significance': False}



code_to_name = {'t': 'Context Independent', 'ct': 'Context Dependent'}

for site in all_sites:
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

    fig, axes = plt.subplots(len(meta['probes_to_plot']), 4, squeeze=False)

    # get a specific probe after a set of different transitions
    for pp, probe in enumerate(meta['probes_to_plot']):


        Z, significance_masks, dpca = cdPCA.tran_dpca(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                                     smooth_window=meta['smoothing_window'], significance=meta['significance'])

        expl_var = dpca.explained_variance_ratio_

        # plots the first PC projection of each context, for each marginalization
        # includes a measurement of significance by shuffle test
        time = np.linspace(0,1,100, endpoint=False)
        bar_bottom = np.zeros(len(expl_var['t']))
        for vv, (marginalization, arr) in enumerate(Z.items()):

            C = arr.shape[1] # n_contexts
            for c in range(C):
                axes[pp, vv].plot(time, arr[0, c, :], label=meta['transitions'][c])

            if meta['significance']:
                if marginalization in significance_masks:

                    left, right = axes[pp, vv].get_xlim()
                    bottom, top = axes[pp, vv].get_ylim()
                    Ychunk = (top-bottom)/10
                    axes[pp, vv].set_ylim(bottom-Ychunk, top)
                    axes[pp, vv].imshow(significance_masks[marginalization][0][None, :],
                                        extent=[0, 1, bottom-Ychunk, bottom], aspect='auto', )
                                    # cmap='gray_r',vmin=0,vmax=1)

            lm = len(Z)-1 #last marginalziation, for labeling purposes
            if vv == lm and pp == 0: axes[pp, vv].legend()

            # ax labels on the leftmost and lower subplots
            if vv == 0: axes[pp, vv].set_ylabel(f'Probe {probe} norm spk rate')

            if pp == len(meta['probes_to_plot'])-1: axes[pp, vv].set_xlabel('time (s)')
            else:axes[pp, vv].set_xticklabels([])

            if pp == 0:
                axes[pp, vv].set_title(f'{code_to_name[marginalization]}\n'
                                       f'1st {code_to_name[marginalization]} component')
            else:
                axes[pp, vv].set_title(f'1st {code_to_name[marginalization]} component')


        # marginalization weights
        weight_ax = axes[pp,2]
        cdPCA.weight_pdf(dpca, marginalization='ct', axes=weight_ax, cellnames=goodcells)

        # plots explained varianceP
        var_ax = axes[pp,3]
        cdPCA.variance_explained(dpca, var_ax)

        fig.suptitle(f'{site}')


    # set figure to full size in tenrec screen
    fig.set_size_inches(19.2, 9.79)

    analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
    root = f'/home/mateo/Pictures/transitions_dPCA/{analysis_parameters}'
    if not os.path.isdir(root): os.mkdir(root)
    fig.savefig(f'{root}/{site}.png', dpi=100)
    #  fig.savefig(f'{root}/{unique_filename}.svg')

