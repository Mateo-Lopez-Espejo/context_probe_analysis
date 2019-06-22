import nems.recording as recording
import nems_lbhb.baphy as nb
import cpp_epochs as cpe

from cpn_triplets import split_recording

"I am lazy, this is a one liner to load a formated cpp/cpn signal"

def load(site, **kwargs):

    options = {'batch': 316,
               'siteid': site,
               'stimfmt': 'envelope',
               'rasterfs': 100,
               'recache': False,
               'runclass': 'CPN',
               'stim': False}  # ToDo chace stims, spectrograms???

    load_URI = nb.baphy_load_recording_uri(**options)
    loaded_rec = recording.load_recording(load_URI)

    CPN_rec = cpe.set_recording_subepochs(loaded_rec)

    recordings  = split_recording(CPN_rec)
    return recordings

recs = load('AMT028b')