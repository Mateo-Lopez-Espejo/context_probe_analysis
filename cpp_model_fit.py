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

# gets test recording, as would be the output of CPP preprocesing

testfile = '/home/mateo/code/context_probe_analysis/pickles/BRT037b'
rec = jl.load(testfile)
# ToDo split into estimation validation sets
est = rec
val = rec

# basic modelspec, weighted channels, linear filter and DC shift
modelspec_name='wc.2x2.g-fir.2x15-lvl.1'

# parses some data from rec meta into the analysis meta
meta = {'modelname': modelspec_name, 'recording': None, **rec.meta}

modelspec = nems.initializers.from_keywords(modelspec_name, meta=meta)



# iterates over each cell, creating a new stim recording containing the response of all other cells in the population

all_cells_phy = list()

for ii, cellid in enumerate(rec.meta['cellid']):

    # does the signal construction for each data set
    for subset in [est, val]:
        stim = subset['stim'].rasterize()
        resp = subset['resp'].rasterize()




    # prefits the dcshift
    modelspec = nems.initializers.prefit_to_target(
            est, modelspec, nems.analysis.api.fit_basic,
            target_module='levelshift',
            fitter=scipy_minimize,
            fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}})