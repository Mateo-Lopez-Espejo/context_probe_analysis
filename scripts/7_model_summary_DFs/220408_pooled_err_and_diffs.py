import itertools as itt
import pathlib as pl
from collections import defaultdict
from configparser import ConfigParser

import joblib as jl
import numpy as np
from tqdm import tqdm

from src.models.modelnames import modelnames
from src.models.param_tools import get_pred_err
from src.root_path import config_path
from src.utils.subsets import cellid_A1_fit_set, cellid_PEG_fit_set, batch_map

"""
Set of metrics comparing multiple model predictions, and their relative relation with one another.
"""

def aggregate_all_differences(modelnicknames, floor, cellids, instances=None):
    """
    aggregate values for each instance of context_pair probe  cell that are significant. both for the real data
    and for the model predictions
    """

    def inner(cid):

        err, diff_err_floor = get_pred_err(cid, batch_map[cid], modelnames[floor], part='probe')
        ctx, prb, tme = err.shape
        del (err)

        out_differences = dict()
        out_errors = dict()
        for mm, nickname in enumerate(modelnicknames):
            _, diff_err, diff_resp, diff_pred = get_pred_err(cid, batch_map[cid], modelnames[nickname], part='probe',
                                                             retur_diffs=True)
            if instances is None:
                diff_pred = diff_pred.reshape((-1, 100))
                if mm == 0:
                    diff_resp = diff_resp.reshape((-1, 100))

                if nickname != floor:
                    diff_err = (diff_err - diff_err_floor).reshape((-1, 100))

            else:
                ctx_pairs = [f'{c0:02}_{c1:02}' for c0, c1 in itt.combinations(range(ctx), 2)]
                view = instances.query(f"id == '{cid}'")
                # create an indexer to slice all significant instances in a single go
                if len(view) == 0:
                    return None
                ctx_idx = list()
                prb_idx = list()
                for rr, row in view.iterrows():
                    ctx_idx.append(ctx_pairs.index(row.context_pair))
                    prb_idx.append(row.probe - 1)

                diff_pred = diff_pred[ctx_idx, prb_idx, :]

                if mm == 0:
                    diff_resp = diff_resp[ctx_idx, prb_idx, :]

                if nickname != floor:
                    diff_err = (diff_err - diff_err_floor)[ctx_idx, prb_idx, :]

            # now that we have our selection, flip values so they are possitive and take means
            out_differences[nickname] = np.where(diff_pred.sum(axis=-1, keepdims=True) < 0, diff_pred * -1,
                                                 diff_pred).mean(axis=0)
            if mm == 0:
                out_differences['response'] = np.where(diff_resp.sum(axis=-1, keepdims=True) < 0,
                                                       diff_resp * -1, diff_resp).mean(axis=0)

            if nickname != floor:
                out_errors[f'{nickname}-{floor}'] = diff_err.mean(axis=0)

        label = cid
        instance_count = diff_pred.shape[0]

        return out_differences, out_errors, label, instance_count

    pooled_differences = defaultdict(list)
    pooled_errors = defaultdict(list)
    labels = list()
    instance_counter = 0

    # parallelized
    output = jl.Parallel(n_jobs=4)(jl.delayed(inner)(cid) for cid in cellids)
    for out in output:

    # # vanila
    # for cid in tqdm(cellids):
    #     out = inner(cid)
        if out is not None:
            out_differences, out_errors, label, instance_count = out
        else:
            continue

        for key, val in out_differences.items():
            pooled_differences[key].append(val)
        for key, val in out_errors.items():
            pooled_errors[key].append(val)

        labels.append(label)
        instance_counter += instance_count

    pooled_differences = {key: np.stack(val, axis=0) for key, val in pooled_differences.items()}
    pooled_errors = {key: np.stack(val, axis=0) for key, val in pooled_errors.items()}

    return pooled_differences, pooled_errors, labels, instance_counter





config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220310_ctx_mod_metric_DF_tstat_cluster_mass_BS'
DF = jl.load(summary_DF_file)

#### lets do it in a for loop
cellids = cellid_A1_fit_set.union(cellid_PEG_fit_set)

file_names = ['220408_pooled_differences_trunc150',
              '220408_pooled_nonsig_differences']
querries = [
    f"metric == 'integral_trunc1.5' and mult_comp_corr == 'bf_cp' and source == 'real' and cluster_threshold == 0.05 and value > 0 and id in {list(cellids)}",
    f"metric == 'integral' and mult_comp_corr == 'bf_cp' and source == 'real' and cluster_threshold == 0.05 and value == 0 and id in {list(cellids)}",
]

for filename, query in zip(file_names, querries):
    cache_file = pl.Path(config['paths']['analysis_cache']) / filename
    recache = False
    if cache_file.exists() and not recache:
        print(f'cache for pooled nonsignificant differences at {cache_file}')
        # pooled_diff_err, labels = jl.load(cache_file)
    else:
        # load the real data to find the subset of neuron_context-pair_probes with significant modulation
        filtered = DF.query(query)

        out = aggregate_all_differences(modelnicknames=['STRF_long_relu', 'pop_lone_relu', 'pop_mod_relu'],
                                        floor='STRF_long_relu',
                                        cellids=cellids,
                                        instances=filtered)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print(out[3])
        jl.dump(out, cache_file)
    pass
