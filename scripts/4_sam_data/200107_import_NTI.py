import nems.recording as recording
import nems_lbhb.baphy as nb
from nems import db as nd
from nti_epochs import set_recording_subepochs

'''
tying to import data from Sam Norman-Heineger. This data comes from experiments with the format NTI, which are a kilt of 
short varied sounds used to calculate integration by his method (code in Matlab, not worth port in into python).

in short, this I should be alble of import the data with the appropriate tags, so I can organized and splice context-probe
pairs to use my integration analysis approach
'''

batch = 319  # NTI batch, Sam paradigm

# check sites in batch
batch_cells = nd.get_batch_cells(batch)
cell_ids = batch_cells.cellid.unique().tolist()
site_ids = set([cellid.split('-')[0] for cellid in cell_ids])

# for site in site_ids:
site = 'AMT028b'
options = {'batch': batch,
           'siteid': site,
           'stimfmt': 'envelope',
           'rasterfs': 100,
           'recache': False,
           'runclass': 'NTI',
           'stim': False}
load_URI = nb.baphy_load_recording_uri(**options)
loaded_rec = recording.load_recording(load_URI)
epochs = loaded_rec.epochs

NTI_rec = set_recording_subepochs(loaded_rec)

'''
script finished, relevant functions have bee wrapped and moved to nti_epochs.py
'''