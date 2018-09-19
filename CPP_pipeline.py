
import nems_db.baphy as nb
import nems.recording as recording
import cpp_epochs as cpe
import joblib as jl
import os
import cpp_plots as cplt
import cpp_PCA as cpca
import matplotlib.pyplot as plt
import numpy as np
import nems.epoch as nep
import cpp_dispersion as disp


# multi electode array loading options
options = {'batch': 310,
           'site': 'BRT037b',
           'rasterfs': 100}

# todo this is creating a cache in charlies directory, move somewhere else!
# gives the uri to a cached recording. if the cached data does not exists, creates it and saves it.

Test = True

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
        ax.set_ylabel('cumulative explained variance');
        ax.set_title('PCA: fraction of variance explained')


    # selects most responsive celll
    scat_key = {'s':5, 'alpha':0.5}
    cplt.hybrid(sig, epoch_names='single', channels='all', start=0, end=3, scatter_kws=scat_key,)

    # selects the stimulus generating the highest response
    cplt.hybrid(sig, epoch_names='single', channels=good_cells_index, start=0, end=3, scatter_kws=scat_key)
    # vocalization 3 generates the clearest response

    # plot of the best cell stim combination
    cplt.hybrid(sig, epoch_names=r'\AC\d_P3', channels=11, start=3, end=6, scatter_kws=scat_key)

    # sanity check of Pre and Post StimSilence=
    cplt.hybrid(sig, epoch_names=r'\AC0_P\d', channels=good_cells_index, start=3, end=6, scatter_kws=scat_key)
    cplt.hybrid(sig, epoch_names=r'\AC\d_P0', channels=good_cells_index, start=3, end=3.5, scatter_kws=scat_key)

    # Initially plots statespace considering the most responsive units and then the PCs:

    traj_kws = {'smoothing': 2, # some mild smoothing
                'downsample': 10, # from 100 to 50 hz
                'rep_scat': False,
                'rep_line': True,
                'mean_scat': False,
                'mean_line': True}


    # trajectory in neuron space of two good cells
    cplt.signal_trajectory(sig, dims=[5, 11], epoch_names=r'\AC\d_P3', _trajectory_kws=traj_kws)

    cplt.signal_trajectory(sig_pca, dims=3, epoch_names=r'\AC\d_P3',_trajectory_kws=traj_kws)
    cplt.signal_trajectory(sig_pca, dims=3, epoch_names='C0_P3',_trajectory_kws=traj_kws)


    cplt.signal_trajectory(sig_pca, dims=3, epoch_names=r'\AC0_P\d',_trajectory_kws=traj_kws)
    cplt.hybrid(sig, epoch_names=r'\AC0_P\d', channels='all', start=3, end=6, scatter_kws=scat_key)

# the previous preliminary plots of state space show a lack of stereotipical trajectoryies, this and the
# inspections of the raster plots show hign spont neurons lacking an stereotipical auditory response.
# This lead me to believe that they might be "false" units from errors in the process of spike sorting.
# all further analysis will be done with only good, reponsive cells.

good_cells_index = [5, 7, 8, 10, 11, 14]
good_cell_names = [chan_name for cc, chan_name in enumerate(sig.chans) if cc in good_cells_index]

filt_sig = sig.extract_channels(good_cell_names)
filt_pca, PCA = cpca.signal_PCA(filt_sig, center=True)


if plot == True:

    # variance explained by PCs
    fig, ax = plt.subplots()
    toplot = np.cumsum(PCA.explained_variance_ratio_)
    ax.plot(toplot, '.-')
    ax.set_xlabel('number of components')
    ax.set_ylabel('cumulative explained variance');
    ax.set_title('PCA: fraction of variance explained')

    # ... so far not more encouragig

    # full scatter pltot
    scat_key = {'s':5, 'alpha':0.5}
    cplt.hybrid(filt_sig, epoch_names='single', channels='all', start=0, end=3, scatter_kws=scat_key)
    cplt.hybrid(filt_sig, epoch_names=r'\AC\d_P4', channels=4, start=3, end=6, scatter_kws=scat_key)

    # trajectory plots
    # 3 best units
    traj_kws = {'smoothing': 2, # some mild smoothing
                'downsample': 10, # from 100 to 50 hz
                'rep_scat': False,
                'rep_line': True,
                'mean_scat': False,
                'mean_line': True}
    cplt.signal_trajectory(filt_sig, dims=[0, 3, 4], epoch_names=r'\AC0_P\d',_trajectory_kws=traj_kws)
    cplt.signal_trajectory(filt_pca, dims=3, epoch_names=r'\AC0_P\d', _trajectory_kws=traj_kws)


# try to calculate de dispersion between contexte for individual cells, adds one cell at a time

































# test matrixes
matrixes = sig.rasterize().extract_epochs(nep.epoch_names_matching(sig.epochs, r'\AC\d_P1'))




