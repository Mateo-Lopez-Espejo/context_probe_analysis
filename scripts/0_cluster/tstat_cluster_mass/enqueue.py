import itertools as itt
import nems.db as nd
from src.utils.subsets import good_sites

##### enqueue.py #####
print('enqueuing jobs')
# python environment where you want to run the job
executable_path = '/auto/users/mateo/miniconda3/envs/cpa_tf/bin/python'
# name of script that you'd like to run
script_path = '/auto/users/mateo/code/context_probe_analysis/scripts/0_cluster/tstat_cluster_mass/script.py'

# Parameters to pass to each job i.e. each function call.
sites = good_sites

# some neurons were added and renamed,and must be reprocecced
sites = {'AMT020a', 'AMT026a', 'AMT028b', 'AMT030a', 'AMT032a', 'ARM017a', 'ARM018a', 'ARM019a', 'ARM029a', 'ARM032a',
         'CRD002a', 'CRD003b', 'CRD004a', 'CRD005b', 'CRD011c', 'CRD012b', 'CRD019b', 'TNC012a', 'TNC017a', 'TNC019a',
         'TNC020a', 'TNC021a', 'TNC023a', 'TNC024a', 'TNC028a', 'TNC029a', 'TNC043a', 'TNC044a', 'TNC045a', 'TNC047a',
         'TNC048a', 'TNC050a', 'TNC051a', 'TNC062a', 'ley070a', 'ley072b', 'ley075b'}


cluster_thresholds = [0.05]
load_fns = ['SC', 'PCA']
montecarlo = 11000
raster_fs = 20

# iterates over every mode, checks what cells have not been fitted with it and runs the fit command.
for nn, (site, clust_thresh, load_fn) in enumerate(itt.product(sites, cluster_thresholds, load_fns)):
    note = f'{site}-{load_fn}-tstat_thresh-{clust_thresh}-cluster_mass-montecarlo_{montecarlo}-raster_fs_{raster_fs}'
    args = [site, clust_thresh, load_fn, montecarlo, raster_fs]
    print(note)
    out = nd.add_job_to_queue(args, note, force_rerun=True,
                              user="mateo", codeHash="master",
                              executable_path=executable_path, script_path=script_path,
                              priority=1, GPU_job=0, reserve_gb=0)

    for oo in out:
        print(oo)

print(f'\nenquueued {nn+1} jobs')

