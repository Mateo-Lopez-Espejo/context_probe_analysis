import nems_db.baphy as nb
import nems.recording as recording
import cpp_epochs as cpe
import joblib as jl
import os
import cpp_plots as cplt
import cpp_PCA as cpca
import matplot

# multi electode array loading options
options = {'batch': 310,
           'site': 'BRT037b'}

# todo this is creating a cache in charlies directory, move somewhere else!
# gives the uri to a cached recording. if the cached data does not exists, creates it and saves it.

Test = True

if Test == True:
    # sets automatic path finding
    this_script_dir = os.path.dirname(os.path.realpath(__file__))
    pickle_path = '{}/pickles'.format(this_script_dir)
    test_rec_path = os.path.normcase('{}/BRT037b'.format(pickle_path))
    # test_rec_path = '/home/mateo/context_probe_analysis/pickles/BRT037b'
    loaded_rec = jl.load(test_rec_path)


else:
    load_URI = nb.baphy_load_multichannel_recording(**options)
    loaded_rec = recording.load_recording(load_URI)


# sets epochs correpsondign to individual sounds as well as context probe pairs
rec = cpe.set_recording_subepochs(loaded_rec, set_pairs=True)
sig = rec['resp']
eps = sig.epochs

# plots full reference i.e. include pre and post stim silence and 5 sounds
psth_kws = {'fs':100, 'start':3, 'end':18,
             'ax':None, 'ci':False, 'y_offset':'auto',
             'plt_kws':None }
cplt.recording_PSTH(rec, epoch_names=r'REFERENCE', signal_names=['resp'], psth_kws=psth_kws)

# plot inividual sounds independent of the context
psth_kws = {'fs':100, 'start':None, 'end':None,
             'ax':None, 'ci':False, 'y_offset':'auto',
             'plt_kws':None }
cplt.recording_PSTH(rec, epoch_names='single', signal_names=['resp'], psth_kws=psth_kws)

# plots individual sound dependent on context
psth_kws = {'fs':100, 'start':None, 'end':None,
             'ax':None, 'ci':False, 'y_offset':'auto',
             'plt_kws':None }
cplt.recording_PSTH(rec, epoch_names='pair', signal_names=['resp'], psth_kws=psth_kws)

# transforms the recording into its PCA equivalent

rec_pca = cpca.recording_PCA(rec, inplace=False)


# plots PCs for each single sound

psth_kws = {'fs':100, 'start':None, 'end':None,
             'ax':None, 'ci':False, 'y_offset':'auto',
             'plt_kws':None }
cplt.recording_PSTH(rec_pca, epoch_names='single', signal_names='all', psth_kws=psth_kws)


# plots the variance explained

sig_pca, PCA_dict = cpca.charlie_PCA(sig)
cum_var = PCA_dict['step']




