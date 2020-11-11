import matplotlib.pyplot as plt
import itertools as itt

from src.data.load import load
from src.metrics.reliability import signal_reliability
from src.metrics import trp_dispersion as ndisp
from src.data.cache import make_cache, get_cache
from src.visualization import fancy_plots as plot
from src.data import rasters as tp

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',
                  '#984ea3', '#999999', '#e41a1c', '#dede00']

meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 20, # ms
        'transitions' : ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot' : [2,3,5,6],
        'significance': False,
        'nonparam_shuffle': 1000}
fs = 100 #FixMe put into meta, chaches will need to be regenerated

analysis_name = 'trans_euc-dist_pop-scram'


analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])


code_to_name = {'t': 'Probe', 'ct': 'Context'}

for site, probe in zip(['AMT029a', 'ley070a'],[5,2]):

    # load and format triplets from a site
    recs = load(site)

    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
    sig = rec['resp'].rasterize()

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # gets the full raster
    full_array, invalid_cp, valid_cp, all_contexts, all_probes = \
        tp.make_full_array(sig, channels=goodcells, smooth_window=meta['smoothing_window'])

    full_array = full_array[..., 100:]

    fig = plt.figure()

    trans_arr = tp._extract_triplets_sub_arr(probes=probe, context_types=meta['transitions'], full_array=full_array,
                                             context_names=all_contexts, probe_names=all_probes, squeeze=False)




    # calculates the old single trial pairwise normalized euclidean distance
    trans_pairs = list(itt.combinations(meta['transitions'], 2))
    trans_coun = 0
    for half in range(2):
        for col in range(3):

            trans_pair = trans_pairs[trans_coun]


            signal_name = f'190709_{site}_P{probe}_ct-marg'

            func_args = {'transitions_array': trans_arr, 'probe_names': [probe], 'context_transitions': trans_pair,
                         'probe_order': [probe], 'trans_order': meta['transitions'],
                         'shuffle_num': meta['nonparam_shuffle'], 'trial_combinations': True}

            shuffled_dispersion_time = make_cache(function=ndisp.transition_pair_comparison_by_trials,
                                                  func_args=func_args,
                                                  classobj_name=signal_name, recache=False,
                                                  cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}')
            real, trans_shuffle, trial_shuffle = get_cache(shuffled_dispersion_time)

            # top of half, context identity shuffle
            if trans_coun == 0 :
                trans_ax = plt.subplot2grid((4, 3), (half * 2, col), rowspan=1, colspan=1, fig=fig)
                ref_ax = trans_ax
            else:
                trans_ax = plt.subplot2grid((4, 3), (half * 2, col), rowspan=1, colspan=1, fig=fig, sharey=ref_ax)

            fig, trans_ax = plot.plot_dist_with_CI(real*fs, [trans_shuffle*fs], ['context'], ['green'],
                                             smp_start=0, smp_end=100, smp_line=None, fs=100, ax=trans_ax)

            # bottom of half, cell trial identity shuffle
            trial_ax = plt.subplot2grid((4, 3), (half*2+1, col), rowspan=1, colspan=1, fig=fig, sharey=ref_ax)
            fig, trial_ax = plot.plot_dist_with_CI(real*fs, [trial_shuffle*fs], ['context'], ['purple'],
                                             smp_start=0, smp_end=100, smp_line=None, fs=100, ax=trial_ax,)



            # Formatting
            trans_ax.tick_params(labelsize='15')
            trans_ax.spines['right'].set_visible(False)
            trans_ax.spines['top'].set_visible(False)
            trans_ax.axes.get_xaxis().set_visible(False)

            trial_ax.tick_params(labelsize='15')
            trial_ax.spines['right'].set_visible(False)
            trial_ax.spines['top'].set_visible(False)

            # distance labels only on leftmost plots
            if col == 0:
                trans_ax.set_ylabel(f'euclidean\ndistance (Hz)', fontsize=20)
                # trial_ax.set_ylabel(f'euclidean distance(Hz)', fontsize=20)
                pass
            else:
                trans_ax.axes.get_yaxis().set_visible(False)
                trial_ax.axes.get_yaxis().set_visible(False)

            trans_ax.set_title(f'{trans_pair}', fontsize=20)

            # time only in bottom plot
            if half == 0:
                trial_ax.axes.get_xaxis().set_visible(False)
            else:
                trial_ax.set_xlabel('time (s)', fontsize=20)

            trans_coun += 1


    fig.suptitle(f'{site} probe {probe} pairwise distance', fontsize=20)

    # set figure to full size in tenrec screen
    fig.set_size_inches(19.2, 9.79)

    # root = pl.Path(f'/home/mateo/Pictures/DAC2')
    # if not root.exists(): root.mkdir(parents=True, exist_ok=True)
    # png = root.joinpath(f'paired_distance_{site}_P{probe}').with_suffix('.png')
    # fig.savefig(png, transparent=True, dpi=100)
    # svg = png = root.joinpath(f'paired_distance_{site}_P{probe}').with_suffix('.svg')
    # fig.savefig(svg, transparent=True)
