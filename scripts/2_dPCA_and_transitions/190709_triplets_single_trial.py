import matplotlib.pyplot as plt
import numpy as np
import itertools as itt
import pathlib as pl

from cpn_load import load
from reliability import signal_reliability
import cpn_dPCA as cdPCA
import cpn_dispersion as ndisp
from cpp_cache import make_cache, get_cache
import fancy_plots as plot
import cpn_triplets as tp
import joblib as jl

from cpp_PCA import PSTH_PCA as pca

'''
Calculates single trial corrected euclidean distance between pairs of transitions using origninal neuron space i.e. no dPCA.
This is done independently for each probe.
'''


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',
                  '#984ea3', '#999999', '#e41a1c', '#dede00']

all_sites = ['ley070a', # good site. A1
             'ley072b', # Primary looking responses with strong contextual effects
             'AMT028b', # good site
             'AMT029a', # Strong response, somehow visible contextual effects
             'AMT030a', # low responses, Ok but not as good
             #'AMT031a', # low response, bad
             'AMT032a'] # great site. PEG

# meta parameter. i.e. editable fields

meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 20, # ms
        'transitions' : ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot' : [2,3,5,6],
        'significance': False,
        'nonparam_shuffle': 1000,
        'include_ctx': True}

analysis_name = 'trans_euc-dist_pop-scram'



analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])


for site in all_sites:
    # load and format triplets from a site
    recs = load(site)

    if len(recs) > 2:
        print (f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
    sig = rec['resp'].rasterize()

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # gets the full raster
    full_array, invalid_cp, valid_cp, all_contexts, all_probes = \
        tp.make_full_array(sig, channels=goodcells, smooth_window=meta['smoothing_window'])

    if meta['include_ctx'] is True:
        pass
    else:
        full_array = full_array[..., 100:]

    fig = plt.figure()
    for pp, probe in enumerate(meta['probes_to_plot']):

        trans_arr = tp.extract_sub_arr(probes=probe, context_types=meta['transitions'], full_array=full_array,
                                    context_names=all_contexts, probe_names=all_probes, squeeze=False)

        # first column, plots the PSTH by transition
        PCax = plt.subplot2grid((4, 7), (pp, 0), rowspan=1, colspan=1, fig=fig)

        if meta['include_ctx'] is True:
            time = np.linspace(-1, 1, 200, endpoint=False)
        else:
            time = np.linspace(0, 1, 100, endpoint=False)

        # PSTHjust for display purpose
        # ToDo checke with normal PCA
        arr = np.squeeze(np.nanmean(trans_arr,axis=2))
        C = arr.shape[0]  # n_contexts
        for c in range(C):
            toplot,_ = pca(arr[c,:,:],center=True)
            PCax.plot(time, toplot[0,:], label=meta['transitions'][c], color=CB_color_cycle[c])


        # formats figures
        if pp == 0: PCax.legend()

        # ax labels on the leftmost and lower subplots
        PCax.set_ylabel(f'Probe {probe} norm spk rate')

        if pp == len(meta['probes_to_plot']) - 1:
            PCax.set_xlabel('time (s)')
        else:
            PCax.set_xticklabels([])

        if pp == 0:
            PCax.set_title(f'first PC')

        # plots the explained variance
        fig.suptitle(f'{site}')


        # calculates the old single trial pairwise normalized euclidean distance
        for tt, trans_pair in enumerate(itt.combinations(meta['transitions'], 2)):

            if meta['include_ctx'] is True:
                signal_name = f'190709_{site}_P{probe}_ctx_included'
            else:
                signal_name = f'190708_{site}_P{probe}_ct-marg'


            func_args = {'transitions_array': trans_arr, 'probe_names': [probe], 'context_transitions': trans_pair,
                         'probe_order': [probe], 'trans_order': meta['transitions'],
                         'shuffle_num': meta['nonparam_shuffle'], 'trial_combinations': True}


            shuffled_dispersion_time = make_cache(function=ndisp.transition_pair_comparison_by_trials,
                                                      func_args=func_args,
                                                      classobj_name=signal_name, recache=False,
                                                      cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}')
            real, shuffled, scrambled = get_cache(shuffled_dispersion_time)

            #define subplots
            PairAx = plt.subplot2grid((4, 7), (pp, tt+1), rowspan=1, colspan=1, fig=fig)

            if meta['include_ctx'] is True:
                fig, ax = plot.plot_dist_with_CI(real, [shuffled], ['p < 0.05'], ['gray'],
                                                 smp_start=0, smp_end=200, smp_line=100, fs=100, ax=PairAx)
            else:
                fig, ax = plot.plot_dist_with_CI(real, [shuffled], ['p < 0.05'], ['gray'],
                                                 smp_start=0, smp_end=100, smp_line=0, fs=100, ax=PairAx)


            # set legend on top right subplot, pair comparison title on top subplot row
            # and time axis label in bottom subplot row,
            if pp == 0 and tt+1 == 6:
                PairAx.legend()

            if pp == 0:
                PairAx.set_title(f'{trans_pair[0]}\n{trans_pair[1]}')

            if pp == 3:
                PairAx.set_xlabel('time (s)')

    # set figure to full size in tenrec screen
    fig.set_size_inches(19.2, 9.79)

    analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
    root = pl.Path(f'/home/mateo/Pictures/{analysis_name}/{analysis_parameters}')
    if not root.exists(): root.mkdir(parents=True, exist_ok=True)
    filepath = root.joinpath(site).with_suffix('.png')
    fig.savefig(filepath, dpi=100)

