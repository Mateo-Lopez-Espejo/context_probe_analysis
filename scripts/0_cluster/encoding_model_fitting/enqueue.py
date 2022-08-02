import nems.db as nd

from src.models.modelnames import modelnames
from src.utils.subsets import cellid_A1_fit_set, cellid_PEG_fit_set

##### enqueue.py #####
print('enqueuing jobs')
# python environment where you want to run the job
executable_path = '/auto/users/mateo/miniconda3/envs/cpa_tf_gpu/bin/python'
# name of script that you'd like to run
script_path = '/auto/users/mateo/code/context_probe_analysis/scripts/0_cluster/encoding_model_fitting/script.py'


# from this old set of models the main ones
# selected = {'STP_STRF1_relu', 'STP_STRF2_relu'}

selected = {'match_STRF', 'match_self', 'match_pop', 'match_full',
            'matchl_STRF', 'matchl_self', 'matchl_pop', 'matchl_full'}
selected_modelnames = [modelname for nickname, modelname in modelnames.items() if nickname in selected]

n_added = 0
n_exists = 0
for cell_set, batch in zip([cellid_A1_fit_set, cellid_PEG_fit_set], [326, 327]):
    out = nd.enqueue_models(list(cell_set), batch, selected_modelnames,user='mateo',
                            executable_path=executable_path, script_path=script_path, force_rerun=False)

    for mm , (qid, msg) in enumerate(out):
        if 'Added new' in msg:
            n_added += 1
        elif 'Resetting existing' in msg:
            n_added += 1
        else:
            n_exists += 1
        print(qid, msg)

print(f'added {n_added} jobs to the queue\nskipped {n_exists} jobs already in queue (not started, running, dead)')