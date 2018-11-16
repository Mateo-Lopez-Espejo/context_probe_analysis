import os
import sys

import nems_db.db as nd

import nems.xforms as xforms

# python environment where you want to run the job
executable_path = '/auto/users/mateo/miniconda3/envs/nemsenv2/bin/python'
# name of script that you'd like to run
script_path = '/auto/users/mateo/context_probe_analysis/cluster_script.py'

# parameters that will be passed to script.
force_rerun = True
batch = 310

# define cellids
batch_cells = nd.get_batch_cells(batch=batch).cellid.tolist()
# batch_cells = ['BRT037b-39-1'] # best cell

# define modelspec_name
modelnames = ['wc.2x2.c-fir.2x15-lvl.1-dexp.1',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1',
              'wc.2x2.c-fir.2x15-lvl.1-stategain.18-dexp.1',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.18-dexp.1']

out = nd.enqueue_models(celllist=batch_cells, batch=batch, modellist=modelnames, user='mateo', force_rerun=True,
                        executable_path=executable_path, script_path=script_path)

for oo in out:
    print(oo)

results_table = nd.get_results_file(batch, modelnames=[modelnames[2]])
preds = []
for cell in batch_cells:
    print(cell)
    p = results_table[results_table['cellid'] == cell]['modelpath'].values[0]
    if os.path.isdir(p):
        xfspec, ctx = xforms.load_analysis(p)
        preds.append(ctx['val'][0])
    else:
        sys.exit('Fit for {0} does not exist'.format(cell))
