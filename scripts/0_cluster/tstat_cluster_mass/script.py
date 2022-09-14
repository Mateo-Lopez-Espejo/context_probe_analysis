import os
import sys

import nems.db as nd
import nems.utils
import nems.utils
from src.metrics.consolidated_tstat import tstat_cluster_mass
from src.data.tensor_loaders import tensor_loaders

if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems.utils.progress_fun = nd.update_job_tick
else:
    queueid = 0

if queueid:
    print("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)

# all parameters are pased as string, ensure proper datatypes
site = sys.argv[1]
cluster_threshold = float(sys.argv[2])
load_fn = sys.argv[3]
montecarlo = int(sys.argv[4])
raster_fs = int(sys.argv[5])

raster_meta = {'reliability': 0.1,  # r value
               'smoothing_window': 0,  # ms
               'raster_fs': raster_fs,  # 20 for even quarters, 30 for old analysis
               'zscore': True,
               'stim_type': 'permutations'}

print(f"running single_cell_dprimes for site {site} with threshold {cluster_threshold} and meta:{raster_meta} ")

# ensures raster is recached
trialR, goodcells = tensor_loaders[load_fn].call(site, recache_rec=True, **raster_meta)
print(goodcells)

# the call method, instead of normal function call, forces recache!
_ = tstat_cluster_mass.call(site, cluster_threshold=cluster_threshold, montecarlo=montecarlo,
                       raster_meta=raster_meta, load_fn=load_fn)

# Mark completed in the queue. Note that this should happen last thing!
# Otherwise the job might still crash after being marked as complete.
if queueid:
    nd.update_job_complete(queueid)
