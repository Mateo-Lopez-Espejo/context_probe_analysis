import nems_db.baphy as nb
import nems.recording as recording
import cpp_epochs as cep
import joblib as jl
import os

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
formated_rec = cep.set_recording_subepochs(loaded_rec, set_pairs=False)

# extracts individual sounds and plots for data quality checking




# transforms the recording into its PCA equivalent

# Now for each individual sounds



