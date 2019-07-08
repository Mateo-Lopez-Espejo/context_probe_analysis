import matplotlib.pyplot as plt
import numpy as np
import itertools as itt
import pathlib as pl

from cpn_load import load
from cpn_reliability import signal_reliability
import cpn_dPCA as cdPCA
import cpn_dispersion as ndisp
from cpp_cache import make_cache, get_cache
import cpp_plots as plot

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


for site in all_sites:
    # load and format triplets from a site
    recs = load(site)

    if len(recs) > 2:
        print (f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
    sig = rec['resp'].rasterize()

    fig = plt.figure()
    for pp, probe in enumerate(meta['probes_to_plot']):

        all_transition_pairs =  itt.combinations(meta['transitions'], 2)

        # calculates response realiability and select only good cells to improve analysis

        r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
        goodcells = goodcells.tolist()


        # runs the dPCA for this individual probe
        Z, significance_masks, dpca = cdPCA.tran_dpca(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                                      smooth_window=meta['smoothing_window'], significance=meta['significance'])

        expl_var = dpca.explained_variance_ratio_
        PCs, C, T = Z['t'].shape

        # first column, plots the contexte dependent marginalization
        PCax = plt.subplot2grid((4, 7), (pp, 0), rowspan=1, colspan=1, fig=fig)

        time = np.linspace(0, 1, 100, endpoint=False)

        arr = Z['ct']
        C = arr.shape[1]  # n_contexts
        for c in range(C):
            PCax.plot(time, arr[0, c, :], label=meta['transitions'][c])

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
            PCax.set_title(f'ct marginalization\nfirst PC')

        # plots the explained variance
        fig.suptitle(f'{site}')


        # calculates the old single trial pairwise normalized euclidean distance
        for tt, tran_pair in enumerate(all_transition_pairs):

            PairAx = plt.subplot2grid((4, 7), (pp, tt+1), rowspan=1, colspan=1, fig=fig)

            signals = cdPCA.signal_transform_triplets_(sig, probe=probe, channels=goodcells,
                                                       smooth_window=meta['smoothing_window'], dpca=dpca)
            trans_sig = signals['ct']

            signal_name = f'190703_{site}_{probe}_ct'

            func_args = {'signal': trans_sig, 'probe_names': [probe], 'context_transitions': list(tran_pair), 'channels': 'all',
                         'shuffle_num': 1000, 'trial_combinations': True}

            analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])

            shuffled_dispersion_time = make_cache(function=ndisp.signal_single_trial_dispersion_pooled_shuffled,
                                                      func_args=func_args,
                                                      classobj_name=signal_name, recache=False,
                                                      cache_folder=f'/home/mateo/mycache/transitions_dPCA/{analysis_parameters}')
            real, shuffled, scrambled = get_cache(shuffled_dispersion_time)

            fig, ax = plot.plot_dist_with_CI(real, shuffled, start=100, end=200, fs=100, ax=PairAx)

            # set pair comparison on top subplot row and time axis in bottom subplot row
            if pp == 0:
                PairAx.set_title(f'{tran_pair[0]}\n{tran_pair[1]}')

            if pp == 3:
                PairAx.set_xlabel('time (s)')

    # set figure to full size in tenrec screen
    fig.set_size_inches(19.2, 9.79)

    analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
    root = pl.Path(f'/home/mateo/Pictures/tran_dPCA_euc/{analysis_parameters}')
    if not root.exists(): root.mkdir(parents=True, exist_ok=True)
    filepath = root.joinpath(site).with_suffix('.png')
    fig.savefig(filepath, dpi=100)

