import itertools as itt
import nems.db as nd
from src.data.load import get_site_ids

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
sites = ('TNC014a',) # test site

# og delayed lines
modelname="ozgf.fs100.ch18-ld.popstate-dline.10.15-norm-epcpn.seq-avgreps_" \
          "dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1-stategain.S.d_" \
          "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

# mean, corrected delayed lines
modelname="ozgf.fs100.ch18-ld.popstate-dline.15.15.1-norm-epcpn.seq-avgreps_" \
          "dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1-stategain.S.d_" \
          "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"

for batch in [326, 327]: # CPN-NTI for A1 and PEG respectively

    valid_neurons = get_site_ids(batch)
    valid_neurons = {ss: nn for ss, nn in valid_neurons.items() if ss in sites}

    for site, cellids in valid_neurons.items():
        for cellid in cellids:
            # if cellid != 'TNC014a-22-2':
            #     continue

            note = f'{cellid}__{modelname}'
            args = [cellid, modelname, batch]
            print(note)
            out = nd.add_job_to_queue(args, note, force_rerun=True,
                                      user="mateo", codeHash="master",
                                      executable_path=executable_path, script_path=script_path,
                                      priority=1, GPU_job=1, reserve_gb=0)
