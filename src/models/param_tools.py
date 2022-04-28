import re
import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import numpy as np
from joblib import Memory

from nems.xform_helper import load_model_xform
from nems.plots.heatmap import _get_wc_coefficients, _get_fir_coefficients

from src.data.rasters import raster_from_sig, load_site_formated_prediction
from src.root_path import config_path

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
xform_memory = Memory(str(pl.Path(config['paths']['recording_cache']) / 'parameters'))

@xform_memory.cache
def load_model_xform_faster(cellid, batch, modelname):
    # TODO there is somethig inside xfspec that cannot be pickled, drop until I figure a better approach
    xfspec, ctx = load_model_xform(cellid=cellid, batch=batch, modelname=modelname, eval_model=True)
    return ctx


def get_population_weights(cellid, batch, modelname, **kwargs):

    if 'ctx' in kwargs.keys():
        # shortcut to avoid time consuming load
        ctx = kwargs['ctx']
    else:
        xfspec, ctx = load_model_xform(cellid=cellid, batch=batch, modelname=modelname, eval_model=False)

    ms = ctx['modelspec']

    # find the postion of the stategain module
    modules = str(ms).split('\n')
    idx = modules.index('nems.modules.state.state_dc_gain')

    _ = ms.set_cell(0)
    _ = ms.set_fit(0)

    chn, npop = ms.phi[idx]['d'].shape

    mean_pop_gain = np.empty((ms.jack_count, npop))

    for jc in range(ms.jack_count):
        _ = ms.set_jack(jc)
        mean_pop_gain[jc, :] = ms.phi[idx]['d'][0,:] # drops cell first singleton dimension

    mean_pop_gain = mean_pop_gain.mean(axis=0)

    return mean_pop_gain

def get_population_influence(cellid, batch, modelname, **kwargs):
    """
    returns the PSTH of the population dependent summed value. i.e. the influence of the population on the PSTH alone
    """
    # runs only preprocecing
    ctx = load_model_xform_faster(cellid=cellid, batch=batch, modelname=modelname)

    mean_pop_gain = get_population_weights(cellid, batch, modelname, ctx=ctx)

    pop_state = ctx['val']['state'].copy()
    pop_state._data = mean_pop_gain[np.newaxis,:] @ pop_state._data

    pop_state.chans = cellid[:7]
    pop_state.nchans = 1


    raster = raster_from_sig(pop_state, probes='all', channels=pop_state.chans[0], contexts='all',
                             smooth_window=0, raster_fs=pop_state.fs,
                             stim_type='permutations',
                             zscore=False, part='all')

    return pop_state, raster


def get_strf(cellid, batch, modelname):
    xfspec, ctx = load_model_xform(cellid=cellid, batch=batch, modelname=modelname, eval_model=False)
    ms = ctx['modelspec']

    # find first instance of sampling frequency modelname
    fs = int(re.findall('\.fs\d*\.', ms.meta['modelname'])[0][3:-1])

    wcc = _get_wc_coefficients(ms, idx=0)
    firc = _get_fir_coefficients(ms, idx=0, fs=fs)

    strf = wcc.T @ firc

    return strf


def get_pred_err(cellid, batch, modelname, part, retur_diffs=False):
    """
    calculats the prediction error, and the contextual difference prediction error across all combinations of context or
    context-pairs and probes.
    """
    ctx = load_model_xform_faster(cellid=cellid, batch=batch, modelname=modelname)

    resp = ctx['val']['resp']
    pred = ctx['val']['pred']
    pred.chans = resp.chans

    raster_resp = raster_from_sig(resp, probes='all', channels=resp.chans[0], contexts='all',
                             smooth_window=0, raster_fs=resp.fs,
                             stim_type='permutations',
                             zscore=False, part=part)


    raster_pred = raster_from_sig(pred, probes='all', channels=pred.chans[0], contexts='all',
                             smooth_window=0, raster_fs=pred.fs,
                             stim_type='permutations',
                             zscore=False, part=part)

    err = ((raster_resp - raster_pred)**2).squeeze(axis=(0,1))

    # now replace context by context-pair derived difference
    rep, chn, ctx, prb, tme = raster_resp.shape
    ctx_pairs = list(itt.combinations(range(ctx), 2))

    diff_resp = np.empty([len(ctx_pairs), prb, tme])
    diff_pred = np.empty_like(diff_resp)

    for cpidx, (c0, c1) in enumerate(ctx_pairs):
        diff_resp[cpidx,:,:] = raster_resp[0,0,c0,:,:] - raster_resp[0,0,c1,:,:]
        diff_pred[cpidx,:,:] = raster_pred[0,0,c0,:,:] - raster_pred[0,0,c1,:,:]

    diff_err = (diff_resp - diff_pred)**2

    if retur_diffs:
        return err, diff_err, diff_resp, diff_pred
    else:
        return err, diff_err


#### more complex prediction comparisons ###

def model_independence_comparison(cellid, batch, independent_models, dependent_model, part):
    site = cellid.split('-')[0]
    rasters = dict()
    modelnames = independent_models + [dependent_model, ]
    for modelname in modelnames:
        raster, goodcells = load_site_formated_prediction(site, modelname=modelname, batch=batch,
                                                                     cellid=cellid, part=part)

        rasters[modelname] = raster.squeeze(axis=(0,1)) # neither multiple repetitions or cells


    # first takes the sum of the indepndente models, then get means over time
    model_sum = rasters[independent_models[0]] + rasters[independent_models[1]]

    # aggregates over time, currently sum. I have tryied variance but there is somethign odd
    sum_agg = model_sum.mean(axis=-1)
    dep_agg = rasters[dependent_model].mean(axis=-1)

    aggs = {'sum':sum_agg, 'dependent': dep_agg}

    return rasters, aggs



if __name__ == '__main__':
    from src.models.modelnames import pop_mod_relu

    cellid = 'TNC014a-22-2'
    batch = 326

    modelname = pop_mod_relu

    # mean_pop_gain = get_population_weights(cellid=cellid, batch=batch, modelname=modelname)
    # mean_pop_gain = get_strf(cellid=cellid, batch=batch, modelname=modelname)
    # mean_pop_gain = get_population_influence(cellid=cellid, batch=batch, modelname=modelname)

    acc, diff_acc = get_pred_err(cellid, batch, modelname, part='probe')
