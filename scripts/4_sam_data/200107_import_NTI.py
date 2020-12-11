import nems.recording as recording
import nems_lbhb.baphy as nb
import collections as coll
from nems import db as nd
from src.data.nti_epochs import set_recording_subepochs

'''
tying to import data from Sam Norman-Heineger. This data comes from experiments with the format NTI, which are a kilt of 
short varied sounds used to calculate integration by his method (code in Matlab, not worth port in into python).

in short, this I should be alble of import the data with the appropriate tags, so I can organized and splice context-probe
pairs to use my integration analysis approach
'''

batch = 318  # NTI in A1
batch = 319  # NTI in PEG

# check sites in batch
batch_cells = nd.get_batch_cells(batch)
cell_ids = batch_cells.cellid.unique().tolist()
site_ids = set([cellid.split('-')[0] for cellid in cell_ids])

# for site in site_ids:
site = 'AMT028b'
options = {'batch': batch,
           'cellid': site,
           'stimfmt': 'envelope',
           'rasterfs': 100,
           'recache': False,
           'runclass': 'NTI',
           'stim': False}
load_URI, _ = nb.baphy_load_recording_uri(**options)
loaded_rec = recording.load_recording(load_URI)
epochs = loaded_rec.epochs

NTI_rec = set_recording_subepochs(loaded_rec)


def get_NTI_files(batch):

    batch_cells = nd.get_batch_cells(batch=batch)

    site_dict = coll.defaultdict(list)
    for cellid in batch_cells.cellid:
        site_dict[cellid[:7]].append(cellid)

    sites_files = list()
    for site, cells in site_dict.items():
        fdb = nd.get_cell_files(cellid=cells[0], runclass='NTI')
        filename = fdb.respfile.values[0].split('.')[0]
        sites_files.append(filename)

    return sites_files
A1 = get_NTI_files(318)
PEG = get_NTI_files(319)
print('\nA1 files\n', A1)
print('\nPEG files\n', PEG)


'''
script finished, relevant functions have bee wrapped and moved to nti_epochs.py
'''