import itertools as itt
import nems.db as nd

from src.models.modelnames import modelnames
from src.utils.subsets import cellid_subset_01, cellid_subset_02, cellid_A1_fit_set, cellid_PEG_fit_set

##### enqueue.py #####
print('enqueuing jobs')
# python environment where you want to run the job
executable_path = '/auto/users/mateo/miniconda3/envs/cpa_tf_gpu/bin/python'
# name of script that you'd like to run
script_path = '/auto/users/mateo/code/context_probe_analysis/scripts/0_cluster/encoding_model_fitting/script.py'


# ###### legacy methods ######
# # full (only inlcudes batch 326), avoids reffiting
# cellids = cellid_A1_fit_set.difference(cellid_subset_01.union(cellid_subset_01)) # this is dangerous since new cells have been added
# eg_already_fig = cellid_subset_01.union(cellid_subset_02)
#
# selected = ['STRF_long_relu', 'self_mod_relu', 'pop_mod_relu', 'self_lone_relu', 'pop_lone_relu']
# modelnames = {nickname:modelname for nickname, modelname in modelnames.items() if nickname in selected}
#
# for batch, (nickname, modelname) in itt.product([326], modelnames.items()): # CPN-NTI for A1 and PEG respectively
#         for batch, cellids  in zip([326], [eg_already_fig]):
#             for cellid in cellids:
#                 note = f'{cellid}_{batch}__{modelname}'
#                 args = [cellid, batch, modelname]
#                 print(note)
#                 out = nd.add_job_to_queue(args, note, force_rerun=True,
#                                           user="mateo", codeHash="master",
#                                           executable_path=executable_path, script_path=script_path,
#                                           priority=1, GPU_job=0, reserve_gb=0)
#



# preffered method, has internal logic to avoid refittign models
selected = ['STRF_long_relu', 'pop_lone_relu', 'pop_mod_relu', 'self_mod_relu', 'self_lone_relu']
selected_modelnames = [modelname for nickname, modelname in modelnames.items() if nickname in selected]

n_added = 0
n_exists = 0
for cell_set, batch in zip([cellid_A1_fit_set, cellid_PEG_fit_set], [326, 327]):
    out = nd.enqueue_models(list(cell_set), batch, selected_modelnames,user='mateo',
                            executable_path=executable_path, script_path=script_path)

    for mm , (qid, msg) in enumerate(out):
        if 'Added new' in msg:
            n_added += 1
        elif 'Resetting existing' in msg:
            n_added += 1
        else:
            n_exists += 1
        print(qid, msg)

print(f'added {n_added} jobs to the queue\nskipped {n_exists} jobs already in queue (not started, running, dead)')

out = nd.enqueue_models(list(cell_set), batch, selected_modelnames, user='mateo',
                        executable_path=executable_path, script_path=script_path)