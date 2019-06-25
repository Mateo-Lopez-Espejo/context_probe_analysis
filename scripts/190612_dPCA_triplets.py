import os

import matplotlib.pyplot as plt
import numpy as np
from dPCA import dPCA

import cpp_plots as cplot

import cpn_triplets as tp
from cpn_load import load
from cpn_reliability import signal_reliability
from cpp_parameter_handlers import _channel_handler

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
        'smoothing_window' : 50, # ms
        'transitions' : ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot' : [2,3,5,6]}



code_to_name = {'t': 'Context Independent', 'ct': 'Context Dependent'}

for site in all_sites:
    # load and format triplets from a site
    # site = 'AMT028b' # good site
    recs = load(site)
    rec = recs['trip0']
    sig = rec['resp'].rasterize()

    # calculates response realiability and select only good cells to improve analysis

    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    if len(goodcells) < 10:
        n_components = len(goodcells)
    elif len(goodcells) == 0:
        continue
    else:
        n_components = 10
    # plots PSTHs of all probes after silence
    # fig, axes = cplot.hybrid(sig, epoch_names=r'\AC0_P[2356]\Z', channels=goodcells)

    # plots PSHTs of individual best probe after all contexts
    # fig, axes = cplot.hybrid(sig, epoch_names=r'\AC\d_P3\Z', channels=goodcells)

    # takes an example probe
    full_array, invalid_cp, valid_cp, all_contexts, all_probes = \
        tp.make_full_array(sig, channels=goodcells, smooth_window=50)

    fig, axes = plt.subplots(len(meta['probes_to_plot']), 3, squeeze=False)

    # get a specific probe after a set of different transitions
    for pp, probe in enumerate(meta['probes_to_plot']):
        # probe = 2
        trialR = tp.extract_sub_arr(probe=probe, context_types=meta['transitions'], full_array=full_array,
                                     context_names=all_contexts, probe_names=all_probes)
        trialR = trialR[:, :, :, 100:] # get only the response to the probe and not the context

        # reorders dimentions from Context x Trial x Neuron x Time  to  Trial x Neuron x Context x Time
        trialR = trialR.transpose([1,2,0,3])
        Tr, N, C, T = trialR.shape
        # trial-average data
        R = np.mean(trialR,0)
        # center data
        R -= np.mean(R.reshape((N,-1)),1)[:,None,None]

        # initializes model
        dpca = dPCA.dPCA(labels='ct',regularizer='auto', n_components=n_components, join={'ct' : ['c','ct']})
        dpca.protect = ['t']

        # Now fit the data (R) using the model we just instantiated. Note that we only need trial-to-trial data when we want to
        # optimize over the regularization parameter.
        Z = dpca.fit_transform(R,trialR)

        # check for significance
        significance_masks = dpca.significance_analysis(R,trialR,axis='t',n_shuffles=100,n_splits=100,n_consecutive=1)

        # get the components that explain the most variation for each marginalization, for some reason they are not ordered?
        top_pc = {key: np.argmax(np.asarray(value)) for key, value in dpca.explained_variance_ratio_.items()}

        # plots the first PC projection of each context, for each marginalization
        # includes a measurement of significance by shuffle test
        time = np.linspace(0,1,T, endpoint=False)
        bar_bottom = np.zeros(n_components)
        for vv, (marginalization, arr) in enumerate(Z.items()):

            for c in range(C):
                axes[pp, vv].plot(time, arr[top_pc[marginalization], c, :], label=meta['transitions'][c])

            if marginalization in significance_masks:

                left, right = axes[pp, vv].get_xlim()
                bottom, top = axes[pp, vv].get_ylim()
                Ychunk = (top-bottom)/10
                axes[pp, vv].set_ylim(bottom-Ychunk, top)
                axes[pp, vv].imshow(significance_masks[marginalization][top_pc[marginalization]][None, :],
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
                                       f'{top_pc[marginalization]+1}st {code_to_name[marginalization]} component')
            else:
                axes[pp, vv].set_title(f'{top_pc[marginalization] + 1}st {code_to_name[marginalization]} component')

            # plots the explained variance
            x = np.arange(n_components) + 1  # the x locations for the groups
            y = np.asarray(dpca.explained_variance_ratio_[marginalization])
            width = 0.35  # the width of the bars: can also be len(x) sequence

            axes[pp,lm+1].bar(x, y, width, bottom=bar_bottom, label=marginalization, color=f'C{9 - vv}')
            axes[pp,lm+1].set_ylabel('explained variance (%)')
            if pp == 0: axes[pp,2].legend()
            bar_bottom += y

        fig.suptitle(f'{site}')


    # set figure to full size in my screen
    fig.set_size_inches(19.2, 9.79)

    unique_filename = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
    unique_filename = f'{site}_{unique_filename}'
    root = '/home/mateo/Pictures/transitions_dPCA'
    if not os.path.isdir(root): os.mkdir(root)
    fig.savefig(f'{root}/{unique_filename}.png', dpi=100)
    #  fig.savefig(f'{root}/{unique_filename}.svg')

