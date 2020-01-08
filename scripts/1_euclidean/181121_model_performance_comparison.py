import collections as col

import joblib as jl
import nems.db as nd
import numpy as np
import pandas as pd

import cpp_dispersion as cdisp
import cpp_epochs as cep
import nems.modelspec as ms
import nems.xforms as xforms
import matplotlib.pyplot as plt
import cpp_plots as cplt
import itertools as itt

batch = 310
results_file = nd.get_results_file(batch)

all_models = results_file.modelname.unique().tolist()
result_paths = results_file.modelpath.tolist()
mod_modelnames = [ss.replace('-', '_') for ss in all_models]

models_shortname = {'wc.2x2.c-fir.2x15-lvl.1-dexp.1': 'LN',
                    'wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1': 'STP',
                    'wc.2x2.c-fir.2x15-lvl.1-stategain.18-dexp.1': 'pop',
                    'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.18-dexp.1': 'STP_pop'}

all_cells = nd.get_batch_cells(batch=310).cellid.tolist()

goodcell = 'BRT037b-39-1'
best_model = 'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.18-dexp.1'

test_path = '/auto/data/nems_db/results/310/BRT037b-39-1/BRT037b-39-1.wc.2x2.c_stp.2_fir.2x15_lvl.1_stategain.18_dexp.1.fit_basic.2018-11-14T093820/'

rerun = False

# compare goodness of fit between models
#   iteratively go trough file
if rerun == True:
    population_metas = list()
    for filepath in result_paths:
        _, ctx = xforms.load_analysis(filepath=filepath, eval_model=True, only=None)
        meta = ctx['modelspecs'][0][0]['meta']
        #   extract important values into a dictionary
        subset_keys = ['cellid', 'modelname', 'r_test', 'r_fit']
        d = {k: v for k, v in meta.items() if k in subset_keys}
        d['r_test'] = d['r_test'][0]
        d['r_fit'] = d['r_fit'][0]
        population_metas.append(d)
    meta_DF = pd.DataFrame(population_metas)
    jl.dump(meta_DF, '/home/mateo/code/context_probe_analysis/pickles/summary_metrics')
else:
    meta_DF = jl.load('/home/mateo/code/context_probe_analysis/pickles/summary_metrics')


def make_tidy(DF, pivot_by=None, more_parms=None, values='value'):
    # todo, make documentation, what is this funcion doing? I remember it being a more complex version of pivot
    # todo implement make tidy by a signle column, it should be easier.
    # todo implement pivot by multiple columns
    # todo move somewhere else, for instance a git repo with usefull funcitons regardless of analysis.
    if pivot_by is None:
        raise NotImplementedError('poke Mateo')

    if more_parms is None:
        more_parms = [col for col in DF.columns if col != values and col != pivot_by]

    # sets relevant  indexes
    more_parms.append(pivot_by)
    indexed = DF.set_index(more_parms)
    # holds only the value column
    indexed = pd.DataFrame(index=indexed.index, data=indexed[values])
    # checks for duplicates
    if indexed.index.duplicated().any():
        raise ValueError("Index contains duplicated entries, cannot reshape")

    # pivots by unstacking, get parameter columns by reseting
    tidy = indexed.unstack([pivot_by]).reset_index(col_level=pivot_by)
    # cleans unnecessary multiindex columns
    tidy.columns = tidy.columns.droplevel(0)
    return tidy



modelnames = meta_DF.modelname.unique().tolist()
max_rval = np.max(meta_DF.loc[:, ['r_test', 'r_fit']].values)

for mod1, mod2 in itt.combinations(modelnames,2):
    ff_modelname = meta_DF.modelname.isin([mod1, mod2])
    filtered = meta_DF.loc[ff_modelname, :]
    pivoted = filtered.pivot(index='cellid', columns='modelname', values='r_test')
    vals_1 = pivoted[mod1].values
    vals_2 = pivoted[mod2].values
    fig, ax = plt.subplots()
    ax.scatter(vals_1, vals_2)
    ax.set_xlim(0, max_rval+0.1)
    ax.set_ylim(ax.get_xlim())
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
    ax.set_xlabel('{}\n{}'.format(models_shortname[mod1], mod1))
    ax.set_ylabel('{}\n{}'.format(models_shortname[mod2], mod2))

'''
The end word of this script is that adding the population filter increases the performance of the model in a
dramatic way, it is indeed the biggest source of predicion power. this leads to question if we are predicting 
just a bunch of correlated noise. Its is worthwile considering the idea of looking at a covariance matrix 
and somehowe contecting those results to the predictive power of the population.
Aditionaly, as a stategain, the population filter does not have any latencies asociated, this is questionable
given that would be biasing the model to capture common input/noise correlation, rather than the recurrencies
that I was set to find in the first place.
Implementing a proper population linear filter is becoming increasingly important. That is my next task. 
'''




