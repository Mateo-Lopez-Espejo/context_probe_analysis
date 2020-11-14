import os
import collections as col

import nems.db as nd
import numpy as np

import nems.modelspec as ms
import nems.xforms as xforms


def _get_result_paths(batch, cellid_list, modelname):

    results_file = nd.get_results_file(batch)
    ff_cellid = results_file.cellid.isin(cellid_list)
    ff_modelname = results_file.modelname == modelname
    result_paths = results_file.loc[ff_cellid & ff_modelname, 'modelpath'].tolist()

    if not result_paths:
        raise ValueError('no cells in cellid_list fit with this model'.format(modelname))
    elif len(result_paths) != len(cellid_list):
        raise ValueError('inconsitent cells in cell_idlist, and in loading paths\n'
                         'cells')

    return result_paths


def reconsitute_rec(batch, cellid_list, modelname):
    '''
    Takes a group of single cell recordings (from cells of a population recording) including their model predictions,
    and builds a recording withe signals containing the responses and predictions of all the cells in the population
    This is to make the recordigs compatible with downstream dispersion analisis or any analysis working with signals
    of neuronal populations
    :param batch: int batch number
    :param cellid_list: [str, str ...] list of cell IDs
    :param modelname: str. modelaname
    :return: NEMS Recording object
    '''

    result_paths = _get_result_paths(batch, cellid_list, modelname)

    cell_resp_dict = dict()
    cell_pred_dict = col.defaultdict()

    for ff, filepath in enumerate(result_paths):
        # use modelsepcs to predict the response of resp
        xfspec, ctx = xforms.load_analysis(filepath=filepath, eval_model=False, only=slice(0, 2, 1))
        modelspecs = ctx['modelspecs'][0]
        cellid = modelspecs[0]['meta']['cellid']
        real_modelname = modelspecs[0]['meta']['modelname']
        rec = ctx['rec'].copy()
        rec = ms.evaluate(rec, modelspecs)  # recording containing signal for resp and pred

        # holds and organizes the raw data, keeping track of the cell for later concatenations.
        cell_resp_dict.update(
            rec['resp']._data)  # in PointProcess signals _data is already a dict, thus the use of update
        cell_pred_dict[cellid] = rec[
            'pred']._data  # in Rasterized signals _data is a matrix, thus the requirement to asign key.

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


def reconstitute_modelspecs(batch, cellid_list, modelname, module='stp'):

    # Todo, right now it extracts only tau and u for the STP module, make it general

    result_paths = _get_result_paths(batch, cellid_list, modelname)

    cell_resp_dict = dict()
    cell_pred_dict = col.defaultdict()

    tau = list()
    u = list()
    cellids = list()

    for ff, filepath in enumerate(result_paths):
        # get the modelspecs for this fit
        mspaths = []
        for file in os.listdir(filepath):
            if file.startswith("modelspec"):
                mspaths.append(filepath + "/" + file)
        ctx = xforms.load_modelspecs([], uris=mspaths, IsReload=False)
        modelspecs = ctx['modelspecs'][0]
        cellid = modelspecs[0]['meta']['cellid']

        # finds the first insntance of the specified module
        for mod in modelspecs:
            mod_id = mod['id'].split('.')[0]
            module = 'stp' # todo delete this override
            if mod_id == module:
                selected_module = mod
                break
        else:
            raise ValueError('this modelname does not contain the specified module')

        tau.append(selected_module['phi']['tau'])
        u.append(selected_module['phi']['u'])
        cellids.append(cellid)

    tau = np.stack(tau, axis=0)
    u = np.stack(u, axis=0)

    reconstituted_modelspecs = {'cellid':cellids, 'tau':tau, 'u':u}

    return reconstituted_modelspecs

'''
import cpp_reconstitute_rec as crec 
modelname = 'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1'
cellid_list = crec.get_site_ids(310)['BRT057b']
rec = crec.reconsitute_rec(310, cellid_list, modelname)
'''
