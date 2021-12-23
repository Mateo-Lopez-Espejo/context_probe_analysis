import collections as col
from configparser import ConfigParser
from joblib import Memory
import pathlib as pl

from nems import db as nd
from nems_lbhb.baphy_experiment import BAPHYExperiment

from src.data.stim_paradigm import split_recording
from src.root_path import config_path

"I am lazy, this is a one liner to load a formated cpp/cpn signal"
config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
memory = Memory(str(pl.Path(config['paths']['recording_cache']) / 'rasters'))

@memory.cache
def load(site, **kwargs):
    # defaults

    options = {'batch': 316,
               'cellid': site,
               'stimfmt': 'envelope',
               'rasterfs': 100,
               'runclass': 'CPN',
               'stim': False,
               'resp':True,
               'recache': False}

    options.update(**kwargs)

    manager = BAPHYExperiment(cellid=site, batch=options['batch'])

    loaded_rec = manager.get_recording(**options)
    parameters = manager.get_baphy_exptparams()

    recordings  = split_recording(loaded_rec, parameters)

    return recordings, parameters

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
