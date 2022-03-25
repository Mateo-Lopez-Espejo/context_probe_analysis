import collections as col
from configparser import ConfigParser
from time import time

import numpy as np
from joblib import Memory
import pathlib as pl

from nems import db as nd
from nems.xform_helper import find_model_xform_file
from nems_lbhb.baphy_experiment import BAPHYExperiment
from nems.signal import RasterizedSignal
from nems.recording import  Recording, load_recording

from src.data.stim_paradigm import split_recording
from src.root_path import config_path

"I am lazy, this is a one liner to load a formated cpp/cpn signal"
config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
resp_memory = Memory(str(pl.Path(config['paths']['recording_cache']) / 'rasters'))
pred_memory = Memory(str(pl.Path(config['paths']['recording_cache']) / 'predictions'))

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

@resp_memory.cache
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



@pred_memory.cache
def load_pred(id, modelspec, batch, **kwargs):
    """
    load an individual cell or all cells in a site and concatenates
    """

    if len(id) == 12:
        # loads single neuron
        cellid = id
        uri, _ = find_model_xform_file(cellid,batch,modelspec)
        uri = pl.Path(uri) / 'prediction.tar.gz'
        rec = load_recording(uri)
        rec['pred'].chans = [cellid]

    elif len(id) == 7:
        # loads all single neurons and concatenate in a new recording
        site = id
        print(f'loading and concatenating all cell predictions in {site}')
        cellids = list(get_site_ids(batch)[site])
        rasters = list()

        tic = time()
        for cellid in cellids:
            print(f'   {cellid}')

            uri, _ = find_model_xform_file(cellid,batch,modelspec)
            uri = pl.Path(uri) / 'prediction.tar.gz'
            rec = load_recording(uri)
            pred = rec['pred']
            fs = pred.fs
            epochs = pred.epochs
            rasters.append(pred._data)
        print(f'it took {time()-tic:.3f}s ')

        rasters = np.concatenate(rasters,axis=0)

        pred = RasterizedSignal(data=rasters, fs=fs, name='pred', recording=site, epochs=epochs, chans=cellids)
        rec = Recording(signals={'pred':pred})

    else:
        raise ValueError(f'cannot interpret id with lenth {len(id)}. Is this a cell or a site?')

    return rec


if __name__ == '__main__':

    cellid = 'TNC014a-22-2'
    siteid = cellid[:7]
    modelname = "ozgf.fs100.ch18-ld.popstate-dline.10.15-norm-epcpn.seq-avgreps_" \
                "dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1-stategain.S.d_" \
                "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

    batch = 326
    rec = load_pred(cellid, modelname, batch)
    print(f"done loading cell with shape{rec['pred'].shape}")
    # rec = load_pred(siteid, modelname, batch)
    # print(f"done loading site with shape{rec['pred'].shape}")


    pass