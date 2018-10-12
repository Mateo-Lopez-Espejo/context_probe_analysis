import joblib as jl
import matplotlib.pyplot as plt
import numpy as np

import cpp_PCA as cpca
import cpp_dispersion as cdisp
import cpp_epochs as cpe
import cpp_plots as cplt
import nems.recording as recording
import nems_db.baphy as nb
import itertools as itt

options = {'batch': 310,
           'site': 'BRT037b',
           'rasterfs': 100}

# todo this is creating a cache in charlies directory, move somewhere else!
# gives the uri to a cached recording. if the cached data does not exists, creates it and saves it.

Test = False

if Test == True:
    # # sets automatic path finding
    # this_script_dir = os.path.dirname(os.path.realpath(__file__))
    # pickle_path = '{}/pickles'.format(this_script_dir)
    # test_rec_path = os.path.normcase('{}/BRT037b'.format(pickle_path))
    test_rec_path = '/home/mateo/context_probe_analysis/pickles/BRT037b'
    loaded_rec = jl.load(test_rec_path)

else:
    load_URI = nb.baphy_load_multichannel_recording(**options)
    loaded_rec = recording.load_recording(load_URI)

# sets eps correpsondign to individual sounds as well as context probe pairs
rec = cpe.set_recording_subepochs(loaded_rec, set_pairs=True)
sig = rec['resp']
eps = sig.epochs

# transforms the recording into its PCA equivalent
rec_pca, pca_stats = cpca.recording_PCA(rec, inplace=False, center=True)
sig_pca = rec_pca['resp_PCs']

plot = False

if plot == True:

    # variance explained by PCs

    for sig_name, pca in pca_stats.items():
        fig, ax = plt.subplots()
        toplot = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(toplot, '.-')
        ax.set_xlabel('number of components')
        ax.set_ylabel('cumulative explained variance')
        ax.set_title('PCA: fraction of variance explained')

    # selects most responsive celll
    scat_key = {'s': 5, 'alpha': 0.5}
    cplt.hybrid(sig, epoch_names='single', channels='all', start=0, end=3, scatter_kws=scat_key, )

    good_cell_index = [5, 7, 8, 10, 11, 14]

    # selects the stimulus generating the highest response
    cplt.hybrid(sig, epoch_names='single', channels=good_cell_index, start=0, end=3, scatter_kws=scat_key)
    # vocalization 3 generates the clearest response

    # plot of the best cell stim combination
    cplt.hybrid(sig, epoch_names=r'\AC\d_P3', channels=11, start=3, end=6, scatter_kws=scat_key)

    # sanity check of Pre and Post StimSilence=
    cplt.hybrid(sig, epoch_names=r'\AC0_P\d', channels=good_cell_index, start=3, end=6, scatter_kws=scat_key)
    cplt.hybrid(sig, epoch_names=r'\AC\d_P0', channels=good_cell_index, start=3, end=3.5, scatter_kws=scat_key)

    # Initially plots statespace considering the most responsive units and then the PCs:

    traj_kws = {'smoothing': 2,  # some mild smoothing
                'downsample': 10,  # from 100 to 50 hz
                'rep_scat': False,
                'rep_line': True,
                'mean_scat': False,
                'mean_line': True}

    # trajectory in neuron space of two good cells
    cplt.signal_trajectory(sig, dims=[5, 11], epoch_names=r'\AC\d_P3', _trajectory_kws=traj_kws)

    cplt.signal_trajectory(sig_pca, dims=3, epoch_names=r'\AC\d_P3', _trajectory_kws=traj_kws)
    cplt.signal_trajectory(sig_pca, dims=3, epoch_names='C0_P3', _trajectory_kws=traj_kws)

    cplt.signal_trajectory(sig_pca, dims=3, epoch_names=r'\AC0_P\d', _trajectory_kws=traj_kws)
    cplt.hybrid(sig, epoch_names=r'\AC0_P\d', channels='all', start=3, end=6, scatter_kws=scat_key)

# the previous preliminary plots of state space show a lack of stereotipical trajectoryies, this and the
# inspections of the raster plots show hign spont neurons lacking an stereotipical auditory response.
# This lead me to believe that they might be "false" units from errors in the process of spike sorting.
# all further analysis will be done with only good, reponsive cells.

good_cell_index = [5, 7, 8, 10, 11, 14]
good_cell_names = [chan_name for cc, chan_name in enumerate(sig.chans) if cc in good_cell_index]

filt_sig = sig.extract_channels(good_cell_names)
filt_pca, PCA = cpca.signal_PCA(filt_sig, center=True)

if plot is True:
    # variance explained by PCs
    fig, ax = plt.subplots()
    toplot = np.cumsum(PCA.explained_variance_ratio_)
    ax.plot(toplot, '.-')
    ax.set_xlabel('number of components')
    ax.set_ylabel('cumulative explained variance')
    ax.set_title('PCA: fraction of variance explained')

    # ... so far not more encouragig

    # full scatter pltot
    scat_key = {'s': 5, 'alpha': 0.5}
    cplt.hybrid(filt_sig, epoch_names='single', channels='all', start=0, end=3, scatter_kws=scat_key)
    cplt.hybrid(filt_sig, epoch_names=r'\AC\d_P4', channels=4, start=3, end=6, scatter_kws=scat_key)

    # trajectory plots
    # 3 best units
    traj_kws = {'smoothing': 2,  # some mild smoothing
                'downsample': 10,  # from 100 to 50 hz
                'rep_scat': False,
                'rep_line': True,
                'mean_scat': False,
                'mean_line': True}
    cplt.signal_trajectory(filt_sig, dims=[0, 3, 4], epoch_names=r'\AC0_P\d', _trajectory_kws=traj_kws)
    cplt.signal_trajectory(filt_pca, dims=3, epoch_names=r'\AC0_P\d', _trajectory_kws=traj_kws)

if plot is True:
    ############################## dispersion analyis
    # example of full population ploting for a set of all contexts to a probe
    epoch_names = r'\AC\d_P3'
    channels = 'all'
    fig, ax = cdisp.plot_single_context(sig, channels, epoch_names)
    fig.suptitle('full population example for probe 3')

    ##############################
    # considers each combination of cell/probe as an independent recording (disregard cell identity)
    # 1 iterates over all relevant probes i.e. excludes silence

    fig, ax = cdisp.pseudopop_significance(sig, channels='all')
    fig.suptitle('all cells * all cpp')
    ax.axvline(3, color='red')

    # sanity check 1. a cell without any significant bins
    insig_eps = r'\AC\d_P4'
    insig_cell = 'BRT037b-39-1'
    cdisp.plot_single_context(sig, insig_cell, insig_eps)

    # sanity check 2. plots the cell with the latest significant bin
    sign_eps = r'\AC\d_P4'
    sign_cell = 'BRT037b-06-1'
    cdisp.plot_single_context(sig, sign_cell, sign_eps)

    ##############################
    # repeat the 'population sumary plots' but only keeping good cells.
    fig, ax = cdisp.pseudopop_significance(sig, channels=good_cell_index)
    fig.suptitle('good cells * all cpps, significant difference over time')
    ax.axvline(3, color='red')

    # plots worst and best
    best_cell = 'BRT037b-46-1'
    best_probe = r'\AC\d_P4'
    cdisp.plot_single_context(sig, best_cell, best_probe)

    worst_cell = 'BRT037b-31-1'
    worst_probe = r'\AC\d_P1'
    cdisp.plot_single_context(sig, worst_cell, worst_probe)

    ##############################
    ### calculates pvalues with a rolling window
    epoch_names = r'\AC\d_P3'
    channels = good_cell_index

    # calculates kurskal wallis between different context across time
    wind_kws = {'window': 5, 'rolling': True, 'type': 'MSD'}

    # plots all cells for example cpp
    fig, ax = cdisp.plot_single_context(sig, channels=channels, epochs=epoch_names, **wind_kws)

    fig, ax = cdisp.pseudopop_significance(sig, channels=channels, **wind_kws)
    ax.axvline(3, color='red')
    fig.suptitle('all cells * all cpp, kruskal window = 5')

    # plots best
    sig_eps = r'\AC\d_P1'
    sig_cell = 'BRT037b-33-3'
    fig, ax = cdisp.plot_single_context(sig, channels=sig_cell, epochs=sig_eps, **wind_kws)
    fig.suptitle('latest significance')

    # plots worst
    insig_eps = r'\AC\d_P4'
    insig_cell = 'BRT037b-38-1'
    fig, ax = cdisp.plot_single_context(sig, channels=insig_cell, epochs=insig_eps, **wind_kws)
    fig.suptitle('earliest significance')

##############################
# uses high fs for the analysis and tries windows of consecutives, of different lengths

if plot is True:
    out = cdisp.pseudopop_significance(sig, channels='all', sign_fs=100, window=1, rolling=True, type='MSD',
                                       consecutives=[1, 2, 3, 4, 5], recache=False, signal_name=None)


    # then plots the equivalent to the multiple comparisons window but downsamplig to the adecuate window size

    for freq in (20, 25, 33, 50):
        out = cdisp.pseudopop_significance(sig, channels='all', sign_fs=freq, window=1, rolling=True, type='MSD',
                                           consecutives=[1], recache=False, signal_name=None)


    # does the same for an example good cell
    sig_eps = r'\AC\d_P3'
    sig_cell = 'BRT037b-39-1'
    fig_list = cdisp.plot_single_context(sig, channels=sig_cell, epochs=sig_eps, sign_fs=100, raster_fs=100, psth_fs=100,
                                         window=1, rolling=True, type='MSD', consecutives=[1, 2, 3, 4, 5])

    # give that the "best looking" significance plot is when considering siginificance for 4 consecutive window
    # an equivalent aproach would be to do the analysis with an equivalent downsampling of 1/4, i.e. 25hz \
    # with the added advantage of reduced calculation time.

    # full population
    out = cdisp.pseudopop_significance(sig, channels=channels, sign_fs=20, window=1, rolling=True, type='MSD',
                                       consecutives=[1])

    # example cell
    sig_eps = r'\AC\d_P3'
    sig_cell = 'BRT037b-39-1'
    fig_list = cdisp.plot_single_context(sig, channels=sig_cell, epochs=sig_eps, sign_fs=20, raster_fs=100, psth_fs=20,
                                         window=1, rolling=True, type='MSD', consecutives=[1])



######### second go into population

disp_mat, cell_names = cdisp.signal_all_context_sigdif(sig, channels=good_cell_names, probes=(1, 2, 3, 4), dimensions='population',
                                                            sign_fs=10, window=1, rolling=True, type='Euclidean', recache=False, value='metric')

print('matrix shape: {}'.format(disp_mat.shape))


# calculates significance from pvalue, plots
significance = cdisp._significance_criterion(disp_mat, axis=1, window=1, threshold=0.5)
fig, ax = plt.subplots()
for ii in range(4):

    # toplot = pop_disp.matrix[ii,:] + ii
    toplot = significance[ii, :] + ii
    label = cell_names[ii]

    ax.plot(toplot, label=label)
    ax.legend()

# plots the metric
fig, ax = plt.subplots()
for ii in range(disp_mat.shape[0]):
    toplot = disp_mat[ii, :] + (ii * 2)
    ax.plot(toplot, label=cell_names[ii])

ax.legend()

##### compares euclidean distance for good and all cells, in a cell by cell or population manner

dimension =['cell', 'population']
channels = [good_cell_names, 'all']

mat_dict = dict()
name_dict = dict()

for dim, chan in itt.product(dimension, channels):
    dist_mat, dist_name = cdisp.signal_all_context_sigdif(sig, channels=chan, probes=(1, 2, 3, 4),
                                                  dimensions=dim, sign_fs=10, window=1, rolling=True,
                                                  type='Euclidean', recache=False, value='metric')

    if chan is good_cell_names: chankey = 'best cells'
    elif chan == 'all': chankey = 'all cells'

    if dim == 'cell': dimkey = 'cell by cell'
    elif dim == 'population': dimkey = 'multidimensional'

    key = '{} euclidean distance, for {}'.format(dimkey, chankey)

    mat_dict[key] = dist_mat
    name_dict[key] = dist_name


if plot is True:
    # plots ray values with
    fig, axes = plt.subplots(2,2)
    axes = np.ravel(axes)

    for ax, (key, val) in zip(axes, mat_dict.items()):
        for jj in range(val.shape[0]):

            # sets the dimention as linestile
            words = key.split(' ')
            if words[0] == 'cell':
                linestyle = '--'
            elif words[0] == 'multidimensional':
                linestyle = '-'
            # set cell subset as color
            if words[-2] == 'best':
                color = 'green'
            elif words[-2] == 'all':
                color = 'orange'

            toplot = val[jj, :] + (jj * 1)

            ax.plot(toplot, linestyle=linestyle, color=color)

            ax.set_title(key)
            ax.set_ylabel('euclidean distance')








    # Mean across probes (population) and probe/cell (cell)
    mean_dict = {key: np.nanmean(val, axis=0) for key, val in mat_dict.items()}
    # plots
    fig, ax = plt.subplots()
    for key, val in mean_dict.items():
        words = key.split(' ')
        # sets the dimention as linestile
        if words[0] == 'cell': linestyle = '--'
        elif words[0] == 'multidimensional': linestyle = '-'
        # set cell subset as color
        if words[-2] == 'best': color = 'green'
        elif words[-2] == 'all': color = 'orange'

        ax.plot(val, linestyle=linestyle, color=color, label=key)

    ax.legend()
    ax.set_ylabel('mean euclidean distance')


    # Normalizes to compare between cell by cell and multidimentional
    norm_dict = {key: val/np.max(val) for key, val in mean_dict.items()}
    # plots
    fig, ax = plt.subplots()
    for key, val in mean_dict.items():
        words = key.split(' ')
        # sets the dimention as linestile
        if words[0] == 'cell':
            linestyle = '--'
        elif words[0] == 'multidimensional':
            linestyle = '-'
        # set cell subset as color
        if words[-2] == 'best':
            color = 'green'
        elif words[-2] == 'all':
            color = 'orange'

        ax.plot(val, linestyle=linestyle, color=color, label=key)

    ax.legend()
    ax.set_ylabel('mean euclidean distance')



