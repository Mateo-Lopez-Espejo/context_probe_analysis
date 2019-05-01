import os
import sys

import nems.db as nd

import nems.xforms as xforms

# python environment where you want to run the job
executable_path = '/auto/users/mateo/miniconda3/envs/nemsenv2/bin/python'
# name of script that you'd like to run
script_path = '/auto/users/mateo/context_probe_analysis/cluster_script.py'

# parameters that will be passed to script.
force_rerun = True
batch = 310

# define modelspec_name
modelnames = ['wc.2x2.c-fir.2x15-lvl.1-dexp.1',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1',
              'wc.2x2.c-fir.2x15-lvl.1-stategain.S-dexp.1',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1']

# define cellids
batch_cells = set(nd.get_batch_cells(batch=batch).cellid)
full_analysis = nd.get_results_file(batch)
already_analyzed = full_analysis.cellid.unique().tolist()
# batch_cells = ['BRT037b-39-1'] # best cell


# iterates over every mode, checks what cells have not been fitted with it and runs the fit command.
for model in modelnames:
    ff_model = full_analysis.modelname == model
    already_fitted_cells = set(full_analysis.loc[ff_model, 'cellid'])

    cells_to_fit = list(batch_cells.difference(already_fitted_cells))

    print('model {}, cells to fit:\n{}'.format(model, cells_to_fit))

    out = nd.enqueue_models(celllist=cells_to_fit, batch=batch, modellist=[model], user='mateo', force_rerun=force_rerun,
                            executable_path=executable_path, script_path=script_path)

    for oo in out:
        print(oo)



DB_pull = False
if DB_pull is True:
    results_table = nd.get_results_file(batch, cellids=list(batch_cells))
    preds = []
    for cell in batch_cells:
        print(cell)
        p = results_table[results_table['cellid'] == cell]['modelpath'].values[0]
        if os.path.isdir(p):
            xfspec, ctx = xforms.load_analysis(p)
            preds.append(ctx['val'][0])
        else:
            sys.exit('Fit for {0} does not exist'.format(cell))
