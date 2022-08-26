from configparser import ConfigParser
from time import time

import numpy as np
import pandas as pd
from joblib import Memory
import pathlib as pl

from nems.db import pd_query, get_batch_cells
from nems.xform_helper import find_model_xform_file
from nems.signal import RasterizedSignal
from nems.recording import  Recording, load_recording

from src.data.stim_structure import split_recording
from src.root_path import config_path


from nems_lbhb.baphy_experiment import BAPHYExperiment


"I am lazy, this is a one liner to load a formated cpp/cpn signal"
config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
resp_memory = Memory(str(pl.Path(config['paths']['recording_cache']) / 'rasters'))
pred_memory = Memory(str(pl.Path(config['paths']['recording_cache']) / 'predictions'))


###### functions to find cells and  sites given conditions ######

def get_batch_ids(batch):
    df = get_batch_cells(batch)
    df['siteid'] = df.cellid.str[:7]
    df.drop(columns='batch', inplace=True)
    return df

def get_runclass_ids(runclass):
    """
    get all cells and sites from a given runclass, e.g. both permutations and triplests from CPN
    """
    querry = f"SELECT gSingleCell.siteid, gSingleCell.cellid FROM " \
             f"((gSingleCell INNER JOIN sCellFile ON gSingleCell.id=sCellFile.singleid) " \
             f"INNER JOIN  gRunClass on sCellFile.runclassid=gRunClass.id) WHERE gRunClass.name = '{runclass}'"
    df = pd_query(querry)
    return df


def get_CPN_ids(nsounds, structure):
    """
    returns a DF of the specified neurons
    :nsounds: int, either 2, (for triplets) or  4 and 10 for permutations
    :structure: str , 'Triplets' or 'AllPermutations'
    """
    #sanitize input
    assert structure in ['Triplets', 'AllPermutations']
    assert nsounds in [2,4,9,10] # that one TNC062 with 9 sounds....

    if structure == 'Triplets' and nsounds !=2:
        print('forcing 2 sound for triplets')
        nsounds = 2

    # finds the raw ids for  AllPermutations experiments
    querry = f"SELECT rawid, name, svalue FROM gData where name in ('Ref_SequenceStructure', 'Ref_SoundIndexes')"
    param_df = pd_query(querry)
    param_df = param_df.pivot_table(index=['rawid'], columns=['name'], values='svalue', aggfunc='first')
    param_df = param_df.dropna().reset_index().copy()
    param_df['Ref_SoundIndexes'] = param_df['Ref_SoundIndexes'].apply(lambda x: [int(s) for s in x[1:-1].split(' ')])
    param_df['nsounds'] = param_df['Ref_SoundIndexes'].apply(lambda x: len(x))
    param_df['Ref_SequenceStructure'] = param_df['Ref_SequenceStructure'].str.strip()
    param_df = param_df.query(f"nsounds == {nsounds} and Ref_SequenceStructure == '{structure}'")
    rawids = tuple(param_df.rawid.unique())

    if len(rawids) == 0:
        print('no sites/neurons found with these specifications')
        return pd.DataFrame()

    # finally finds the subset of sites/cells that fulfill all conditions
    querry = f"SELECT cellid, id FROM gDataRaw where id in {rawids}" #this returns a site, not a cell.
    raw_site_df =pd_query(querry).rename(columns={'cellid':'siteid', 'id':'rawid'})
    raw_site_df = raw_site_df.loc[~raw_site_df.siteid.str.contains('tst'),:]

    # for some reason this bad marked experiment got passed and saved. delete it
    raw_site_df = raw_site_df.query("rawid != 134847")


    # takes all CPN data and filters first by selected site/rawid then adds sound indices based on rawid
    df = get_runclass_ids('CPN').merge(
        raw_site_df, on='siteid', validate='m:1'
    ).merge(param_df.loc[:,['Ref_SoundIndexes','rawid']],on='rawid', validate='m:1')

    return df


###### actual loading  functions ######

# @resp_memory.cache(ignore=['recache']) # This cache is mostly for working from home but does not add significant efficiency
def load(site, recache=False, **kwargs):
    # defaults

    options = {'batch': 316,
               'cellid': site,
               'stimfmt': 'envelope',
               'rasterfs': 100,
               'runclass': 'CPN',
               'stim': False,
               'resp':True}

    options.update(**kwargs)
    options['recache'] = recache

    manager = BAPHYExperiment(cellid=site, batch=options['batch'])

    loaded_rec = manager.get_recording(**options)
    parameters = manager.get_baphy_exptparams()

    recordings  = split_recording(loaded_rec, parameters)

    return recordings, parameters



# @pred_memory.cache # This cache is mostly for working from home but does not add significant efficiency
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
        cellids = get_batch_ids(batch).query(f"siteid == {site}").sort_values(ascending=True).tolist()
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
    # rec = load_pred(cellid, modelname, batch)
    rec = load(siteid)
    # print(f"done loading cell with shape{rec['pred'].shape}")
    # rec = load_pred(siteid, modelname, batch)
    # print(f"done loading site with shape{rec['pred'].shape}")
    # out = get_runclass_ids('CPN')
    # out1 = get_CPN_ids(10, 'AllPermutations')
    # out2 = get_CPN_ids(2, 'Triplets')
    # get_batch_ids(326)

    pass