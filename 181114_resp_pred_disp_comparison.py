import collections as col

import joblib as jl
import nems_db.db as nd
import numpy as np
import pandas as pd

import cpp_dispersion as cdisp
import cpp_epochs as cep
import nems.modelspec as ms
import nems.xforms as xforms
import matplotlib.pyplot as plt
import cpp_plots as cplt


batch = 310
results_file = nd.get_results_file(batch)

all_models = results_file.modelname.unique().tolist()
result_paths = results_file.modelpath.tolist()
mod_modelnames = [ss.replace('-', '_') for ss in all_models]

models_shortname = {'wc.2x2.c_fir.2x15_lvl.1_dexp.1': 'LN',
                    'wc.2x2.c_stp.2_fir.2x15_lvl.1_dexp.1': 'STP',
                    'wc.2x2.c_fir.2x15_lvl.1_stategain.18_dexp.1': 'pop',
                    'wc.2x2.c_stp.2_fir.2x15_lvl.1_stategain.18_dexp.1': 'STP_pop'}

all_cells = nd.get_batch_cells(batch=310).cellid.tolist()

goodcell = 'BRT037b-39-1'
best_model = 'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.18-dexp.1'
test_path = '/auto/data/nems_db/results/310/BRT037b-39-1/BRT037b-39-1.wc.2x2.c_stp.2_fir.2x15_lvl.1_stategain.18_dexp.1.fit_basic.2018-11-14T093820/'

rerun = False
# using single cell recording predictions, generates a population recording
if rerun == True:
    pop_rec_dict = col.defaultdict()
    for model in mod_modelnames:

        cell_resp_dict = dict()
        cell_pred_dict = col.defaultdict()

        for cellid in all_cells:
            filepath = [ff for ff in result_paths if cellid in ff and model in ff][0]

            #   use modelsepcs to predict the response of resp
            xfspec, ctx = xforms.load_analysis(filepath=filepath, eval_model=True, only=None)
            modelspecs = ctx['modelspecs'][0]
            cellid = modelspecs[0]['meta']['cellid']
            modelname = modelspecs[0]['meta']['modelname']
            rec = ctx['rec'].copy()
            rec = ms.evaluate(rec, modelspecs)  # recording containing signal for resp and pred
            cell_resp_dict.update(rec['resp']._data)
            cell_pred_dict[cellid] = rec['pred']._data

        # create a new population recording. pull stim from last single cell, create signal from meta form last resp signal and
        # stacked data for all cells. modify signal metadata to be consistent with the previous
        pop_resp = rec['resp']._modified_copy(data=cell_resp_dict,
                                              chans=list(cell_resp_dict.keys()),
                                              nchans=len(list(cell_resp_dict.keys())))

        stack_data = np.concatenate(list(cell_pred_dict.values()), axis=0)
        pop_pred = rec['pred']._modified_copy(data=stack_data,
                                              chans=list(cell_pred_dict.keys()),
                                              nchans=len(list(cell_pred_dict.keys())))

        pop_rec = rec.copy()

        pop_rec['resp'] = pop_resp
        pop_rec['pred'] = pop_pred
        del pop_rec.signals['state']
        del pop_rec.signals['state_raw']

        # stores recordign in a dictionary, renaming with a shorter model name
        pop_rec_dict[model] = pop_rec
    jl.dump(pop_rec_dict, '/home/mateo/code/context_probe_analysis/pickles/pop_predictions')

else:
    pop_rec_dict = jl.load('/home/mateo/code/context_probe_analysis/pickles/pop_predictions')
# renames if necesary
pop_rec_dict = {models_shortname[key]:val for key, val in pop_rec_dict.items()}

################################
# compare the dispersion for the actual response and the predicted response for all models
# format expochs for all recordings
formated = {key: cep.set_recording_subepochs(rec) for key, rec in pop_rec_dict.items()}

# some PSTH plotting to visually asses the quality of the different model fits.
cells = ['BRT037b-39-1']
# cells = 'all'
cells = ['BRT037b-31-1', 'BRT037b-33-3', 'BRT037b-36-3', 'BRT037b-38-1', 'BRT037b-39-1', 'BRT037b-46-1'] # best cells
epoch = r'\AC\d_P4' # best stimulus

for key, rec in formated.items():
    fig, axes = cplt.hybrid(rec['pred'], epoch_names=epoch, channels=cells, start=3, end=6, sub_types=[False, True, False])
    fig.suptitle(key)
fig, axes = cplt.hybrid(formated['LN']['resp'], epoch_names=epoch, channels=cells, start=3, end=6, sub_types=[True, True, False])
fig.suptitle('resp')

# calculatest the dispersion for each prediction and one response
dispersions = {modelname:
                   cdisp.signal_all_context_sigdif(rec['pred'], channels='all', probes=(1, 2, 3, 4),
                                                   dimensions='population', sign_fs=100, window=1, rolling=True,
                                                   type='Euclidean', recache=False,
                                                   signal_name='181115-{}'.format(modelname),
                                                   value='metric')[0]
               for modelname, rec in formated.items()}

dispersions['resp'] = cdisp.signal_all_context_sigdif(formated['LN']['resp'], channels='all', probes=(1, 2, 3, 4),
                                                      dimensions='population', sign_fs=100, window=1, rolling=True,
                                                      type='Euclidean', recache=False,
                                                      signal_name='181115-{}'.format('resp'),
                                                      value='metric')[0]
# caluculatese the pvalue of the dispersion
# calculatest the dispersion for each prediction and one response
pvals = {modelname:
                   cdisp.signal_all_context_sigdif(rec['pred'], channels='all', probes=(1, 2, 3, 4),
                                                   dimensions='population', sign_fs=100, window=1, rolling=True,
                                                   type='Euclidean', recache=False,
                                                   signal_name='181115-{}'.format(modelname),
                                                   value='pvalue')[0]
               for modelname, rec in formated.items()}

pvals['resp'], stim_names = cdisp.signal_all_context_sigdif(formated['LN']['resp'], channels='all', probes=(1, 2, 3, 4),
                                                      dimensions='population', sign_fs=100, window=1, rolling=True,
                                                      type='Euclidean', recache=False,
                                                      signal_name='181115-{}'.format('resp'),
                                                      value='pvalue')




# plots the dispersion over time
fig, axes = plt.subplots(3,2)
axes = np.ravel(axes)
for ii, (key, value) in enumerate(dispersions.items()):
    ax = axes[ii]
    ax.imshow(value, aspect='auto', origin='lower')
    ax.set_title(key)
    ax.set_yticks([0,1,2,3], minor=False)
    ax.set_yticklabels(stim_names)
    ax.axvline(300, color='red')
fig.suptitle('euclidean distance over time for for eache probe')

# plot difference pval over time
fig, axes = plt.subplots(3,2)
axes = np.ravel(axes)
for ii, (key, value) in enumerate(pvals.items()):
    ax = axes[ii]
    ax.imshow(value, aspect='auto', origin='lower')
    ax.set_title(key)
    ax.set_yticks([0,1,2,3], minor=False)
    ax.set_yticklabels(stim_names)
    ax.axvline(300, color='red')
fig.suptitle('pvalue (of euclidean) over time for for eache probe')

# plots significant difference over time
significnaces = {key: cdisp._significance_criterion(val, axis=1, window=1, threshold=0.05, comp='<=')
                 for key, val in pvals.items()}
fig, axes = plt.subplots(3,2)
axes = np.ravel(axes)
for ii, (key, value) in enumerate(significnaces.items()):
    ax = axes[ii]
    ax.imshow(value, aspect='auto', origin='lower', cmap='binary')
    ax.set_title(key)
    ax.set_yticks([0,1,2,3], minor=False)
    ax.set_yticklabels(stim_names)
    ax.axvline(300, color='red')
fig.suptitle('significant difference over time for for eache probe')

# plots all signals (resp, and model predictions) side by side. Collapses all different probes by the mean

collapsed = {key: np.nanmean(val, axis=0) for key, val in dispersions.items()}
# fits an exponential decay. Todo make it work
fitted = {key: cdisp.disp_exp_decay(val, start=300, prior=1, axis=None)[0] for key, val in collapsed.items()}

fig, ax = plt.subplots()
for ii, (key, val) in enumerate(collapsed.items()):
    color = 'C{}'.format(ii)
    ax.plot(val, label=key, color=color)
    ax.plot(dispersions[key].T, color=color, alpha=0.2)
    ax.plot (fitted[key], color=color)
ax.axvline(300, color='black')
ax.legend()
