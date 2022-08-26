import nems.db as nd

from src.models.modelnames import modelnames
from src.utils.subsets import cellid_A1_fit_set, cellid_PEG_fit_set
from src.data.region_map import region_map

##### enqueue.py #####
print('enqueuing jobs')
# python environment where you want to run the job
executable_path = '/auto/users/mateo/miniconda3/envs/cpa_tf_gpu/bin/python'
# name of script that you'd like to run
script_path = '/auto/users/mateo/code/context_probe_analysis/scripts/0_cluster/encoding_model_fitting/script.py'


# from this old set of models the main ones
selected = {'matchl_STRF', 'matchl_self', 'matchl_pop', 'matchl_full'}
selected_modelnames = [modelname for nickname, modelname in modelnames.items() if nickname in selected]

cells_to_refit = {'TNC023a-018-1', 'TNC023a-021-1', 'TNC014a-31-3', 'TNC016a-61-2', 'TNC017a-30-3', 'TNC024a-027-3',
                  'TNC016a-38-2', 'TNC023a-012-1', 'TNC017a-49-5', 'TNC018a-07-4', 'TNC024a-027-1', 'TNC023a-054-2',
                  'TNC017a-49-4', 'TNC023a-003-1', 'TNC023a-033-1', 'TNC016a-55-2', 'TNC023a-054-1', 'TNC017a-31-3',
                  'TNC024a-038-2', 'TNC018a-16-3', 'TNC023a-035-2', 'TNC023a-034-4', 'TNC024a-037-1', 'TNC018a-11-1',
                  'TNC017a-44-4', 'TNC023a-014-1', 'TNC017a-22-2', 'TNC024a-042-2', 'TNC016a-40-5', 'TNC023a-020-1',
                  'TNC023a-023-1', 'TNC024a-042-1', 'TNC023a-025-1', 'TNC024a-026-2', 'TNC017a-41-3', 'TNC024a-031-1',
                  'TNC023a-050-2', 'TNC016a-43-3', 'TNC024a-030-1', 'TNC017a-58-2', 'TNC014a-25-5', 'TNC023a-058-1',
                  'TNC017a-48-2', 'TNC015a-07-1', 'TNC023a-049-1', 'TNC017a-57-2', 'TNC015a-55-2', 'TNC023a-031-1',
                  'TNC023a-020-3', 'TNC017a-10-5', 'TNC023a-045-1', 'TNC023a-034-1', 'TNC024a-024-1', 'TNC010a-031-3',
                  'TNC023a-022-1', 'TNC017a-54-5', 'TNC024a-028-4', 'TNC023a-032-1', 'TNC017a-60-2', 'TNC023a-043-1',
                  'TNC024a-016-1', 'TNC017a-52-2', 'TNC017a-42-2', 'TNC017a-38-1', 'TNC023a-034-2', 'TNC023a-039-1',
                  'TNC023a-046-1', 'TNC023a-030-2', 'TNC024a-034-1', 'TNC017a-25-4', 'TNC023a-050-1', 'TNC017a-57-3',
                  'TNC023a-035-1', 'TNC017a-07-2', 'TNC017a-11-1', 'TNC015a-16-5', 'TNC023a-027-1', 'TNC023a-042-1',
                  'TNC018a-42-6', 'TNC023a-030-1', 'TNC016a-46-2', 'TNC024a-016-2', 'TNC024a-016-3', 'TNC024a-028-1',
                  'TNC024a-040-1', 'TNC023a-055-2', 'TNC017a-10-4', 'TNC018a-05-4', 'TNC024a-028-2', 'TNC024a-022-1',
                  'TNC024a-004-2', 'TNC023a-006-1', 'TNC017a-20-2', 'TNC023a-038-1', 'TNC023a-063-1', 'TNC024a-028-3',
                  'TNC024a-021-1', 'TNC014a-36-6', 'TNC024a-043-1', 'TNC024a-012-1', 'TNC023a-029-2', 'TNC017a-19-5',
                  'TNC024a-015-1', 'TNC017a-24-2', 'TNC023a-051-1', 'TNC015a-50-1', 'TNC015a-58-2', 'TNC023a-055-1',
                  'TNC017a-50-1', 'TNC017a-28-2', 'TNC017a-41-2', 'TNC023a-026-1', 'TNC017a-19-4', 'TNC023a-048-2',
                  'TNC017a-51-3', 'TNC010a-027-3', 'TNC016a-43-7', 'TNC017a-34-3', 'TNC023a-033-2', 'TNC018a-10-2',
                  'TNC024a-027-2', 'TNC014a-41-2', 'TNC010a-040-1', 'TNC024a-026-1', 'TNC010a-007-4', 'TNC023a-036-1',
                  'TNC017a-45-2', 'TNC017a-30-2', 'TNC010a-049-1', 'TNC018a-21-3', 'TNC017a-28-3', 'TNC017a-54-4',
                  'TNC024a-030-2', 'TNC017a-56-1', 'TNC018a-35-1', 'TNC023a-020-2', 'TNC017a-19-6', 'TNC017a-52-3',
                  'TNC024a-038-1', 'TNC017a-31-4', 'TNC017a-45-1', 'TNC023a-057-1', 'TNC017a-09-2', 'TNC024a-043-2',
                  'TNC023a-016-1', 'TNC024a-008-1', 'TNC023a-029-1', 'TNC023a-034-3', 'TNC023a-048-1', 'TNC023a-022-2',
                  'TNC024a-027-4', 'TNC023a-044-1', 'TNC010a-039-1', 'TNC023a-004-1', 'TNC016a-28-5', 'TNC023a-061-1',
                  'TNC023a-052-1', 'TNC017a-51-2', 'TNC016a-27-1', 'TNC024a-004-1', 'TNC018a-54-2', 'TNC023a-037-1',
                  'TNC017a-54-3'}

n_added = 0
n_exists = 0
for cell_set, batch in zip([cellid_A1_fit_set, cellid_PEG_fit_set], [326, 327]):

    cell_set = cell_set.intersection(cells_to_refit)

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
