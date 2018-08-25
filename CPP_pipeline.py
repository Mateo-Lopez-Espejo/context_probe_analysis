import nems_db.baphy as nb
import nems.recording as recording
import cpp_epochs as cep


# multi electode array loading options
options = {'batch': 310,
           'site': 'BRT037b'}

# todo this is creating a cache in charlies directory, move somewhere else!
# gives the uri to a cached recording. if the cached data does not exists, creates it and saves it.
load_URI = nb.baphy_load_multichannel_recording(**options)
# there must be a bette way of seting subepochs!


loaded_rec = recording.load_recording(load_URI)

formated_rec = cep.set_recording_subepochs(loaded_rec)