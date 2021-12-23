import logging
import os
import pathlib as pl
import sys

import joblib as jl

import nems.db as nd
import nems.utils


# test function, to be replaced with whatever to be runned.
def test_cluster(iter):
    from dPCA import dPCA
    testpath = pl.Path(f'/auto/users/mateo/cluster_test/test_{iter}')
    testpath.parent.mkdir(parents=True, exist_ok=True)
    print(f'writing {iter} to {str(testpath)}')

    jl.dump(f'hola test {iter}', testpath)


log = logging.getLogger(__name__)

if __name__ == '__main__':
    if 'QUEUEID' in os.environ:
        queueid = os.environ['QUEUEID']
        nems.utils.progress_fun = nd.update_job_tick
    else:
        queueid = 0

    if queueid:
        print("Starting QUEUEID={}".format(queueid))
        nd.update_job_start(queueid)

    iter = sys.argv[1]

    print(f"Running test_cluster with {iter}")
    out = test_cluster(iter)

    # Mark completed in the queue. Note that this should happen last thing!
    # Otherwise the job might still crash after being marked as complete.
    if queueid:
        nd.update_job_complete(queueid)

else:

    # python environment where you want to run the job
    executable_path = '/auto/users/mateo/miniconda3/envs/context_probe_analysis2/bin/python'
    # name of script that you'd like to run
    script_path = '/auto/users/mateo/cluster_template.py'

    # iterates over every mode, checks what cells have not been fitted with it and runs the fit command.
    for iter in range(5):
        note = f'mateo_test_{iter}'
        args = [iter]

        out = nd.add_job_to_queue(args, note, force_rerun=True,
                                  user="mateo", codeHash="master",
                                  executable_path=executable_path, script_path=script_path,
                                  priority=1, GPU_job=0, reserve_gb=0)

        for oo in out:
            print(oo)


    ### now try to load stuff
    outpath = pl.Path('/auto/users/mateo/cluster_test/') / 'test_0'

    print(jl.load(outpath))
