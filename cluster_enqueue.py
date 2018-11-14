import itertools as itt

import nems_db.db as nd

# python environment where you want to run the job
executable_path = '/auto/users/mateo/miniconda3/envs/nemsenv2/bin/python'
# name of script that you'd like to run
script_path = '/auto/users/mateo/context_probe_analysis/cluster_script.py'

# parameters that will be passed to script.
force_rerun = True
batch = 310

# define cellids
batch_cells = nd.get_batch_cells(batch=batch).cellid
batch_cells = ['BRT37b-39-1'] # best cell

# define modelspec_name
modelnames = ['wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1']
modelnames = ['wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1']
modelnames = ['wc.2x2.c-fir.2x15-lvl.1-dexp.1']

# only old cells without jitter status
# batch_cells = [cellid for cellid in batch_cells if cellid[0:3] != 'gus']



out = nd.enqueue_models(celllist=batch_cells, batch=batch, modellist=modelnames, user='mateo', force_rerun=True,
                  executable_path=executable_path,script_path=script_path)

for oo in out:
    print(oo)


#
# for cellid, modelspec_name in itt.product(batch_cells, modelnames):
#     qid, msg = nd.enqueue_single_model(cellid=cellid, batch=batch, modelname=modelspec_name, user='Mateo',
#                                        force_rerun=force_rerun, executable_path=executable_path,
#                                        script_path=script_path)
#     print(msg)
