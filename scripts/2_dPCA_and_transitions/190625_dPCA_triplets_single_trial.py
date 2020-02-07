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

'''
performs dPCA for context transitions and calculates single trial corrected euclidean distance between pairs of transitions
using single trials projected onto the 'ct' marginalization. i.e. the context dependent variation marginalization.
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
        'nonparam_shuffle': 1000}
####
analysis_name = 'trans_dPCA_euc-dist_pop-scram'
analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])

####
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

    fig = plt.figure()
    for pp, probe in enumerate(meta['probes_to_plot']):

        # runs the dPCA for this individual probe
        Z, trialZ, significance_masks, dpca = cdPCA.tran_dpca(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                                      smooth_window=meta['smoothing_window'], significance=meta['significance'])
        expl_var = dpca.explained_variance_ratio_
        PCs, C, T = Z['t'].shape

        # first column, plots the contexte dependent marginalization
        PCax = plt.subplot2grid((4, 7), (pp, 0), rowspan=1, colspan=1, fig=fig)

        time = np.linspace(0, 1, 100, endpoint=False)

        arr = Z['ct']
        C = arr.shape[1]  # n_contexts
        for c in range(C):
            PCax.plot(time, arr[0, c, :], label=meta['transitions'][c], color=CB_color_cycle[c])

        if meta['significance']:
            if 'ct' in significance_masks:
                left, right = PCax.get_xlim()
                bottom, top = PCax.get_ylim()
                Ychunk = (top - bottom) / 10
                PCax.set_ylim(bottom - Ychunk, top)
                PCax.imshow(significance_masks['ct'][0][None, :],
                            extent=[0, 1, bottom - Ychunk, bottom], aspect='auto', )

        # formats figures
        if pp == 0: PCax.legend()

        # ax labels on the leftmost and lower subplots
        PCax.set_ylabel(f'Probe {probe} norm spk rate')

        if pp == len(meta['probes_to_plot']) - 1:
            PCax.set_xlabel('time (s)')
        else:
            PCax.set_xticklabels([])

        if pp == 0:
            PCax.set_title(f'context first dPC')

        # plots the explained variance
        fig.suptitle(f'{site}')


        # calculates the old single trial pairwise normalized euclidean distance
        for tt, trans_pair in enumerate(itt.combinations(meta['transitions'], 2)):
            # gets the indicese of the different transitions, based on the list of transisions to analyse

            # reshapez Z from PC x C x T into my tranditional shape: C x P x R x PC x T
            tran_arr = np.expand_dims(np.moveaxis(trialZ['ct'], 2,0), 1)

            signal_name = f'190708_{site}_P{probe}_ct-marg'

            func_args = {'transitions_array': tran_arr, 'probe_names': [probe], 'context_transitions': trans_pair,
                         'probe_order': [probe], 'trans_order': meta['transitions'],
                         'shuffle_num': meta['nonparam_shuffle'], 'trial_combinations': True}


            shuffled_dispersion_time = make_cache(function=ndisp.transition_pair_comparison_by_trials,
                                                      func_args=func_args,
                                                      classobj_name=signal_name, recache=False,
                                                      cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}')
            real, cont_shuff, pop_shuff = get_cache(shuffled_dispersion_time)

            #define subplots
            PairAx = plt.subplot2grid((4, 7), (pp, tt+1), rowspan=1, colspan=1, fig=fig)

            fig, ax = plot.plot_dist_with_CI(real, [cont_shuff], ['p < 0.05'], ['gray'],
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

