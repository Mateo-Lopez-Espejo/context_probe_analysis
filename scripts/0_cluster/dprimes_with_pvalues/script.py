import os
import sys

import nems.db as nd
import nems.utils
import nems.utils
from src.metrics.consolidated_dprimes import single_cell_dprimes

if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems.utils.progress_fun = nd.update_job_tick
else:
    queueid = 0

if queueid:
    print("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)

site = sys.argv[1]


meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'stim_type': 'permutations',
        'alpha': 0.05}

print(f"running single_cell_dprimes for site {site} with meta:{meta} ")


_ = single_cell_dprimes(site, contexts='all', probes='all', meta=meta)

# Mark completed in the queue. Note that this should happen last thing!
# Otherwise the job might still crash after being marked as complete.
if queueid:
    nd.update_job_complete(queueid)
