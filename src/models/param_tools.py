import re
import pathlib as pl
from configparser import ConfigParser

import numpy as np
from joblib import Memory

from nems.xform_helper import load_model_xform
from nems.plots.heatmap import _get_wc_coefficients, _get_fir_coefficients

from src.data.rasters import raster_from_sig
from src.root_path import config_path

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
param_memory = Memory(str(pl.Path(config['paths']['recording_cache']) / 'parameters'))


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

@param_memory.cache
def get_population_influence(cellid, batch, modelname, **kwargs):
    # runs only preprocecing
    xfspec, ctx = load_model_xform(cellid=cellid, batch=batch, modelname=modelname,
                                   eval_model=True,)

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


if __name__ == '__main__':
    cellid = 'TNC014a-22-2'
    batch = 326

    modelname = "ozgf.fs100.ch18-ld.popstate-dline.15.15.1-norm-epcpn.seq-avgreps_" \
                "dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1-stategain.S.d_" \
                "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

    # mean_pop_gain = get_population_weights(cellid=cellid, batch=batch, modelname=modelname)
    # mean_pop_gain = get_strf(cellid=cellid, batch=batch, modelname=modelname)
    mean_pop_gain = get_population_influence(cellid=cellid, batch=batch, modelname=modelname)

    import src.data.rasters


    import matplotlib.pyplot as plt
    plt.plot(mean_pop_gain)
    plt.show()


