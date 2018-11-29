import collections as col

import joblib as jl
import numpy as np

import nems.modelspec as ms
import nems.xforms as xforms
import nems_db.db as nd
import collections as col
import pandas as pd

def get_site_ids(batch):
    '''
    returns a list of the site ids for all experiments of a given batch. This site ID helps finding all the cells within
    a population recorded simultaneusly
    :param batch:
    :return:
    '''
    results_file = nd.get_results_file(batch)

    cellids = results_file.cellid.unique().tolist()
    site_IDs = col.defaultdict(list)
    for cell in cellids:
        site_ID = cell.split('-')[0]
        site_IDs[site_ID].append(cell)

    return  dict(site_IDs)


def reconsitute_rec(batch, cellid_list, modelname):

    # get the filename of the fitted model for all the cells in the list
    results_file = nd.get_results_file(batch)
    ff_cellid = results_file.cellid.isin(cellid_list)
    ff_modelname = results_file.modelname == modelname
    result_paths = results_file.loc[ff_cellid & ff_modelname, 'modelpath'].tolist()

    if not result_paths:
        raise ValueError('no cells in cellid_list fit with this model'.format(modelname))
    elif len(result_paths) != len(cellid_list):
        raise ValueError('inconsitent cells in cell_idlist, and in loading paths\n'
                         'cells')

    cell_resp_dict = dict()
    cell_pred_dict = col.defaultdict()

    for ff, filepath in enumerate(result_paths):
        print('cell {}, file {}'.format(ff, filepath))
        # use modelsepcs to predict the response of resp
        # ToDo, there must be a better way of loading the recording and modelspecs to make a prediction
        xfspec, ctx = xforms.load_analysis(filepath=filepath, eval_model=False, only=slice(0,2,1))
        modelspecs = ctx['modelspecs'][0]
        cellid = modelspecs[0]['meta']['cellid']
        real_modelname = modelspecs[0]['meta']['modelname']
        rec = ctx['rec'].copy()
        rec = ms.evaluate(rec, modelspecs)  # recording containing signal for resp and pred

        # holds and organizes the raw data, keeping track of the cell for later concatenations.
        cell_resp_dict.update(rec['resp']._data)  # in PointProcess signals _data is already a dict, thus the use of update
        cell_pred_dict[cellid] = rec['pred']._data  # in Rasterized signals _data is a matrix, thus the requirement to asign key.

    # create a new population recording. pull stim from last single cell, create signal from meta form last resp signal and
    # stacked data for all cells. modify signal metadata to be consistent with new data and cells contained
    pop_resp = rec['resp']._modified_copy(data=cell_resp_dict,
                                          chans=list(cell_resp_dict.keys()),
                                          nchans=len(list(cell_resp_dict.keys())))

    stack_data = np.concatenate(list(cell_pred_dict.values()), axis=0)
    pop_pred = rec['pred']._modified_copy(data=stack_data,
                                          chans=list(cell_pred_dict.keys()),
                                          nchans=len(list(cell_pred_dict.keys())))

    reconstituted_recording = rec.copy()

    reconstituted_recording['resp'] = pop_resp
    reconstituted_recording['pred'] = pop_pred
    del reconstituted_recording.signals['state']
    del reconstituted_recording.signals['state_raw']

    return reconstituted_recording



'''
import cpp_reconstitute_rec as crec 
modelname = 'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1'
cellid_list = crec.get_site_ids(310)['BRT057b']
rec = crec.reconsitute_rec(310, cellid_list, modelname)
'''