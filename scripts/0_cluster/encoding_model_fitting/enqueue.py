import itertools as itt
import nems.db as nd

from src.models.modelnames import modelnames
from src.utils.subsets import cellid_subset_01, cellid_subset_02, cellid_fit_set

##### enqueue.py #####
print('enqueuing jobs')
# python environment where you want to run the job
executable_path = '/auto/users/mateo/miniconda3/envs/cpa_tf_gpu/bin/python'
# name of script that you'd like to run
script_path = '/auto/users/mateo/code/context_probe_analysis/scripts/0_cluster/encoding_model_fitting/script.py'

# Parameters to pass to each job i.e. each function call.

# subset of sites with both NTI and CPN-10 permutationse
sites = {'TNC018a', 'TNC017a', 'TNC023a', 'TNC014a', 'TNC015a', 'TNC010a', 'TNC016a', 'TNC024a', 'TNC009a'}
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a', 'TNC010a'} # empirically decided
sites = sites.difference(badsites)
sites = ('TNC014a',) # test site 1

# top, avoids reffiting the old wrong top
cellids = cellid_subset_02.difference(cellid_subset_01)

# full (only inlcudes batch 326), avoids reffiting
cellids = cellid_fit_set.difference(cellid_subset_01.union(cellid_subset_01))

selected = ['STRF_long_relu', 'self_mod_relu', 'pop_mod_relu', 'self_lone_relu', 'pop_lone_relu']
modelnames = {nickname:modelname for nickname, modelname in modelnames.items() if nickname in selected}

for batch, (nickname, modelname) in itt.product([326], modelnames.items()): # CPN-NTI for A1 and PEG respectively

    # valid_neurons = get_site_ids(batch)
    # valid_neurons = {ss: nn for ss, nn in valid_neurons.items() if ss in sites}

    # for site, cellids in valid_neurons.items():
        for cellid in cellids:
            # if cellid != 'TNC014a-22-2':
            #     continue

            note = f'{cellid}_{batch}__{modelname}'
            args = [cellid, batch, modelname]
            print(note)
            out = nd.add_job_to_queue(args, note, force_rerun=True,
                                      user="mateo", codeHash="master",
                                      executable_path=executable_path, script_path=script_path,
                                      priority=1, GPU_job=0, reserve_gb=0)
