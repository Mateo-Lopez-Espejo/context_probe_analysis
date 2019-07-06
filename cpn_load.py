import nems.recording as recording
import nems_lbhb.baphy as nb
import cpp_epochs as cpe
import re

from cpn_triplets import split_recording
from cpn_reliability import signal_reliability
from cpp_plots import hybrid

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

def plot(cell_or_site, probe, type='perm0', reliability=0.1):

    # matches cell name e.g  'ley070a'
    if re.match(r'\A[a-zA-Z]{3}\d{3}[a-z]',cell_or_site):
        single_cell = True
        site = cell_or_site
    # matches site name e.g. 'ley070a-01-1'
    elif re.match(r'\A[a-zA-Z]{3}\d{3}[a-z]-\d{2}-\d'):
        single_cell = False
        site = cell_or_site[0:7]

    else:
        raise ValueError('wrong format for cell or site')


    signal = load(site)[type]['resp']

    if single_cell:
        goodcells = [cell_or_site]
    else:
        r, goodcells = signal_reliability(signal,'\AC\d_P\d\Z', threshold=reliability)

    # plots PSHTs of individual best probe after all contexts
    fig, axes = hybrid(signal, epoch_names=f'\AC\d_P{probe}\Z', channels=goodcells)




