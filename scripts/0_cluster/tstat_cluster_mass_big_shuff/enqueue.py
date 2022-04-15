import itertools as itt
import nems.db as nd
from src.data.load import get_site_ids

##### enqueue.py #####
print('enqueuing jobs')
# python environment where you want to run the job
executable_path = '/auto/users/mateo/miniconda3/envs/cpa_tf/bin/python'
# name of script that you'd like to run
script_path = '/auto/users/mateo/code/context_probe_analysis/scripts/0_cluster/tstat_cluster_mass_big_shuff/script.py'

# Parameters to pass to each job i.e. each function call.

sites = set(get_site_ids(316).keys())
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a', 'TNC010a'} # empirically decided
no_perm = {'ley058d'} # sites without permutations
sites = sites.difference(badsites).difference(no_perm)
# sites = ('AMT021b',) # test site
# cluster_thresholds = [0.05, 0.01] # the more astringent threshold is perhaps not necesary
cluster_thresholds = [0.05]

# iterates over every mode, checks what cells have not been fitted with it and runs the fit command.
for nn, (site, clust_thresh) in enumerate(itt.product(sites, cluster_thresholds)):
    note = f'{site}_tstat_thresh-{clust_thresh}_cluster_mass_BS'
    args = [site, clust_thresh]
    print(note)
    out = nd.add_job_to_queue(args, note, force_rerun=True,
                              user="mateo", codeHash="master",
                              executable_path=executable_path, script_path=script_path,
                              priority=1, GPU_job=0, reserve_gb=0)

    for oo in out:
        print(oo)

print(f'\nenquueued {nn+1} jobs')

# #cheks if it worked
# from src.metrics.consolidated_dprimes import single_cell_dprimes_cluster_mass
# meta = {'reliability': 0.1,  # r value
#         'smoothing_window': 0,  # ms
#         'raster_fs': 30,
#         'montecarlo': 1000,
#         'zscore': True,
#         'stim_type': 'permutations',
#         'alpha': 0.05}
# out = single_cell_dprimes_cluster_mass('ARM021b', contexts='all', probes='all',
#                                        cluster_threshold=1, meta=meta)
