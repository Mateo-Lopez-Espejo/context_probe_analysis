import os
import sys

import nems.db as nd
import nems.utils
import nems.utils
from nems.xform_helper import fit_model_xform


if 'QUEUEID' in os.environ:
    queueid = os.environ['QUEUEID']
    nems.utils.progress_fun = nd.update_job_tick
else:
    queueid = 0

if queueid:
    print("Starting QUEUEID={}".format(queueid))
    nd.update_job_start(queueid)

cellid = sys.argv[1]
batch = sys.argv[2]
modelname = sys.argv[3]

_ = fit_model_xform(cellid, batch, modelname, autoPlot=True, saveInDB=True, returnModel=False)

# Mark completed in the queue. Note that this should happen last thing!
# Otherwise the job might still crash after being marked as complete.
if queueid:
    nd.update_job_complete(queueid)
