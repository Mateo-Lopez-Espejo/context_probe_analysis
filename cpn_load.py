import collections as col
import nems.recording as recording
import nems_lbhb.baphy as nb
import cpp_epochs as cpe
from cpn_triplets import split_recording
from nems import db as nd

"I am lazy, this is a one liner to load a formated cpp/cpn signal"

def load(site, remote=False, **kwargs):

    # defaults
    options = {'batch': 316,
               # 'siteid': site,
               'cellid': site,
               'stimfmt': 'envelope',
               'rasterfs': 100,
               'recache': False,
               'runclass': 'CPN',
               'stim': False}  # ToDo chace stims, spectrograms???

    options.update(**kwargs)

    if remote is True:
        _, options = nb.parse_cellid(options)
        options = nb.fill_default_options(options)
        d = nd.get_batch_cell_data(batch=options['batch'], cellid=site, label='parm',
                                   rawid=options['rawid'])

        mfilename = list(set(list(d['parm'])))
        mfilename.sort()
        mfilename = ['H:/' + '/'.join(file.split('/')[3:]) for file in mfilename]
        options['mfilename'] = mfilename

    load_URI = nb.baphy_load_recording_uri(**options)
    loaded_rec = recording.load_recording(load_URI)

    CPN_rec = cpe.set_recording_subepochs(loaded_rec)

    recordings  = split_recording(CPN_rec)
    return recordings

# load tests:
# recs = load('AMT028b')


def get_site_ids(batch):
    '''
    returns a list of the site ids for all experiments of a given batch. This site ID helps finding all the cells within
    a population recorded simultaneusly
    :param batch:
    :return:
    '''
    batch_cells = nd.get_batch_cells(batch)

    cellids = batch_cells.cellid.unique().tolist()
    site_IDs = col.defaultdict(list)
    for cell in cellids:
        site_ID = cell.split('-')[0]
        site_IDs[site_ID].append(cell)

    return dict(site_IDs)



# This is just to cache recordings of later http API load

# meta = {'reliability': 0.1,  # r value
#         'smoothing_window': 0,  # ms
#         'raster_fs': 100,
#         'transitions': ['silence', 'continuous', 'similar', 'sharp'],
#         'montecarlo': 1000,
#         'zscore': False}
#
# sites = list(get_site_ids(316).keys())
# for site in sites:
#     recs = load(site, remote=True, rasterfs=meta['raster_fs'], recache=False)
