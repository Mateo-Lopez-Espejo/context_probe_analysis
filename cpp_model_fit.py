import os
import logging
import pandas as pd
import pickle
import nems
import nems.initializers
import nems.priors
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.analysis.api
import nems.utils
import nems.uri
import nems.recording as recording
from nems.signal import RasterizedSignal
from nems.fitters.api import scipy_minimize
import joblib as jl
import numpy as np

# gets test recording, as would be the output of CPP preprocesing

testfile = '/home/mateo/code/context_probe_analysis/pickles/BRT037b'
# testfile = 'C:\\Users\\Mateo\\Science\\David_lab\\code\\context_probe_analysis\\pickles\\BRT037b'  # path for blade
rec = jl.load(testfile)
# ToDo split into estimation validation sets
subsets = {'est': rec, 'val': rec}

# basic modelspec, weighted channels, linear filter and DC shift
modelspec_name = 'wc.2x2.g-fir.2x15-lvl.1'
modelspec_name = 'fir.19x15-lvl.1'


# iterates over each cell, creating a new stim recording containing the response of all other cells in the population
ntime_modspecs = list()

for ii, cellid in enumerate(rec.meta['cellid']):

    working_sets = dict()
    # does the recording construction for each data set
    for key, subset in subsets.items():
        stim = subset['stim'].rasterize()
        resp = subset['resp'].rasterize()

        # splits the response into the cell to be fitted and the cells to use as predictors
        this_cell_resp = np.expand_dims(resp._data[ii, :], 0)
        other_cells_resp = np.delete(resp._data, ii, axis=0)

        # makes the new stmulus array by concatenating stimulus and other cell in the population
        nstim_data = np.concatenate([stim._data, other_cells_resp], axis=0)

        # build signals with modified _data
        stim = stim._modified_copy(data=nstim_data)
        resp = resp._modified_copy(data=this_cell_resp)

        signals = {'stim': stim, 'resp': resp}

        # makes into a recording
        mod_rec = recording.Recording(signals)

        # adds metadata, e
        this_cell_meta = rec.meta.copy()
        this_cell_meta['cellid'] = [cellid]
        mod_rec.meta = this_cell_meta

        # orders in etimation validation sets
        working_sets[key] = mod_rec

    est = working_sets['est']
    val = working_sets['val']

    # parses some data from rec meta into the analysis meta
    analysis_meta = {'modelname': modelspec_name, 'recording': None, **this_cell_meta}
    modelspec = nems.initializers.from_keywords(modelspec_name, meta=analysis_meta)

    # prefits the dc_shift
    modelspec = nems.initializers.prefit_to_target(est,modelspec, nems.analysis.api.fit_basic,
                                                   target_module='levelshift',
                                                   fitter=scipy_minimize,
                                                   fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}})

    # then fit full nonlinear model
    modelspecs = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

    # ----------------------------------------------------------------------------
    # GENERATE SUMMARY STATISTICS

    # generate predictions
    est, val = nems.analysis.api.generate_prediction(est, val, modelspecs)

    # evaluate prediction accuracy
    modelspecs = nems.analysis.api.standard_correlation(est, val, modelspecs)

    ntime_modspecs.append(modelspecs)

    print("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
        modelspecs[0][0]['meta']['r_fit'][0],
        modelspecs[0][0]['meta']['r_test'][0]))



