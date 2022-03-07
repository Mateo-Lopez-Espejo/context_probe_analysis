import os
import sys

import nems.db as nd
import nems.utils
import nems.utils
from src.metrics.consolidated_tstat import single_cell_tstat_cluster_mass

if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems.utils.progress_fun = nd.update_job_tick
else:
    queueid = 0

if queueid:
    print("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)

site = sys.argv[1]
cluster_threshold = float(sys.argv[2])


meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'stim_type': 'permutations'}

print(f"running single_cell_dprimes for site {site} with threshold {cluster_threshold} and meta:{meta} ")


_ = single_cell_tstat_cluster_mass(site, contexts='all', probes='all',
                                     cluster_threshold=cluster_threshold, meta=meta)

# Mark completed in the queue. Note that this should happen last thing!
# Otherwise the job might still crash after being marked as complete.
if queueid:
    nd.update_job_complete(queueid)
