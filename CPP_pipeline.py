import nems_db.baphy as nb
import nems.recording as recording
import cpp_epochs as cpe
import joblib as jl
import os
import cpp_plots as cplt
import cpp_PCA as cpca
import matplotlib.pyplot as plt
import numpy as np


# multi electode array loading options
options = {'batch': 310,
           'site': 'BRT037b',
           'rasterfs': 10}

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

# plots full reference i.e. include pre and post stim silence and 5 sounds
# psth_kws = {'fs':100, 'start':3, 'end':18,
#              'ax':None, 'ci':False, 'y_offset':'auto',
#              'plt_kws':None }
# cplt.recording_PSTH(rec, epoch_names=r'REFERENCE', signal_names=['resp'], psth_kws=psth_kws)


# plot inividual sounds independent of the context
psth_kws = {'fs':10, 'start':None, 'end':None,
             'ax':None, 'ci':False, 'y_offset':'auto',
             'plt_kws':None }
cplt.recording_PSTH(rec, epoch_names='single', signal_names=['resp'], psth_kws=psth_kws)


# plots individual sound dependent on context
# psth_kws = {'fs':100, 'start':None, 'end':None,
#              'ax':None, 'ci':False, 'y_offset':'auto',
#              'plt_kws':None }
# cplt.recording_PSTH(rec, epoch_names='pair', signal_names=['resp'], psth_kws=psth_kws)



# transforms the recording into its PCA equivalent
rec_pca, pca_stats = cpca.recording_PCA(rec, inplace=False, center=True)


# plots PCs for each single sound

psth_kws = {'fs':10, 'start':None, 'end':None,
             'ax':None, 'ci':False, 'y_offset':'auto',
            'channels': 3,
             'plt_kws':None }
cplt.recording_PSTH(rec_pca, epoch_names='single', signal_names='all', psth_kws=psth_kws)


# plots the variance explained for each
for sig_name, pca in pca_stats.items():
    fig, ax = plt.subplots()
    toplot = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(toplot, '.-')
    ax.set_xlabel('number of components')
    ax.set_ylabel('cumulative explained variance');
    ax.set_title()

# plots neuronal trajectory for an example probe within all different contexts

traj_kws = {'smoothing': 1,
            'rep_scat': False,
            'rep_line': True,
            'mean_scat': False,
            'mean_line': True}
cplt.recording_trajectory(rec_pca, dims=3, epoch_names=r'\AC\d_P1', signal_names='PCA', _trajectory_kws=traj_kws)
cplt.recording_trajectory(rec_pca, dims=2, epoch_names=['PreStimSilence'], signal_names='PCA', _trajectory_kws=traj_kws)


# looks at single cell and signle PCA raters for a single probe given multiple contexts



# calcualtes the dispersion between












