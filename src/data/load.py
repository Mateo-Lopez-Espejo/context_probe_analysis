import collections as col
from configparser import ConfigParser
import joblib as jl
import pathlib as pl
from src.data.cache import set_name
import nems.recording as recording
import nems_lbhb.baphy as nb
from nems_lbhb.baphy_experiment import BAPHYExperiment
from src.data import epochs as cpe
from src.data.stim_paradigm import split_recording
from nems import db as nd

"I am lazy, this is a one liner to load a formated cpp/cpn signal"

config = ConfigParser()
config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))


def load(site, boxload=True, **kwargs):

    # defaults
    options = {'batch': 316,
               'cellid': site,
               'stimfmt': 'envelope',
               'rasterfs': 100,
               'recache': False,
               'runclass': 'CPN',
               'stim': False}

    options.update(**kwargs)

    if boxload is True:
        toname = options.copy()
        del toname['recache']
        filename = pl.Path(config['paths']['recording_cache']) / set_name(toname)

        if not filename.parent.exists():
            filename.parent.mkdir()

        if filename.exists() and options['recache'] is False:
            print('loading recording from box')
            loaded_rec = jl.load(filename)

        elif filename.exists() is False or options['recache'] is True:
            load_URI, _ = nb.baphy_load_recording_uri(**options)
            loaded_rec = recording.load_recording(load_URI)
            print('cacheing recoring in box')
            jl.dump(loaded_rec, filename)
        else:
            raise SystemError('WTF?')

    elif boxload is False:
        load_URI, _ = nb.baphy_load_recording_uri(**options)
        loaded_rec = recording.load_recording(load_URI)

    else:
        raise ValueError('boxload must be boolean')

    CPN_rec = cpe.set_recording_subepochs(loaded_rec)

    recordings  = split_recording(CPN_rec)
    return recordings

def load_with_parms(site, **kwargs):
    # defaults

    options = {'batch': 316,
               'cellid': site,
               'stimfmt': 'envelope',
               'rasterfs': 100,
               'runclass': 'CPN',
               'stim': False,
               'resp':True}

    options.update(**kwargs)

    manager = BAPHYExperiment(siteid=site, batch=options['batch'])

    loaded_rec = manager.get_recording(recache=True, **options)
    parameters = manager.get_baphy_exptparams()

    # load_URI, _ = nb.baphy_load_recording_uri(**options)
    # loaded_rec = recording.load_recording(load_URI)

    CPN_rec = cpe.set_recording_subepochs(loaded_rec)
    recordings  = split_recording(CPN_rec)

    return recordings, parameters


# load tests:
# recs = load('AMT028b', recache=True)
# recs, params = load_with_parms('AMT028b')

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



# This is just to check cached recordings of later http API load
# sites = list(get_site_ids(316).keys())
# print(list(sites))
# for site in sites:
#     recs = load(site, rasterfs=30, recache=False)
