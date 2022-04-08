from configparser import ConfigParser
import itertools as itt

from tqdm import tqdm
import numpy as np
import joblib as jl
import pathlib as pl

from src.models.modelnames import modelnames
from src.models.param_tools import get_pred_err
from src.root_path import config_path

"""
Set of metrics comparing multiple model predictions, and their relative relation with one another.
"""

def aggregate_all_errors(modelnicknames, floor, cellids, batch, instances=None):
    """
    aggregate prediciont for all fitted neurons, taking the mean for each neuron and relating the model error prediction
    to the STRF model prediction
    """
    def inner(cid):
        if floor is not None:
            err, diff_err_floor = get_pred_err(cid, batch, modelnames[floor], part='probe')
        ctx, prb, tme = err.shape
        del(err)
        mod_dif_err = np.empty((len(modelnicknames), 100))
        for mm, nickname in enumerate(modelnicknames):
            _, diff_err = get_pred_err(cid, batch, modelnames[nickname], part='probe')

            if instances is None:
                if floor is not None:
                    diff_err = np.mean(diff_err - diff_err_floor, axis=(0, 1))
                else:
                    diff_err = np.mean(diff_err, axis=(0,1))
                mod_dif_err[mm,:] = diff_err
                label = cid
            else:
                ctx_pairs = [f'{c0:02}_{c1:02}' for c0, c1 in itt.combinations(range(ctx), 2)]
                view = instances.query(f"id == '{cid}'")

                #create an indexer to slice all significant instances in a single go
                ctx_idx = list()
                prb_idx = list()
                for rr, row in view.iterrows():
                    ctx_idx.append(ctx_pairs.index(row.context_pair))
                    prb_idx.append(row.probe-1)

                if floor is not None:
                    mod_dif_err[mm, :] = np.mean(diff_err[ctx_idx, prb_idx, :] - diff_err_floor[ctx_idx, prb_idx, :], axis=(0))
                else:
                    mod_dif_err[mm, :] = diff_err[ctx_idx, prb_idx, :].mean(axis=0)

                label = cid

        return mod_dif_err, label

    # # parallelized method
    # output = jl.Parallel(n_jobs=4)(jl.delayed(inner)(cid) for cid in cellids)
    # pooled_diff_err, labels = zip(*output)
    # del output
    # pooled_diff_err = np.stack(pooled_diff_err, axis=0)

    # vanilla method
    pooled_diff_err = list()
    labels = list()
    for cid in tqdm(cellids):
        diff_err, label = inner(cid)
        pooled_diff_err.append(diff_err)
        labels.append(label)

    pooled_diff_err = np.stack(pooled_diff_err, axis=0)
    return pooled_diff_err, labels


def aggregate_all_differences(modelnicknames, cellids, batch, instances=None):
    """
    aggregate values for each instance of context_pair probe  cell that are significant. both for the real data
    and for the model predictions
    """
    def inner(cid):

        model_differences = np.empty((len(modelnicknames), 100))
        for mm, nickname in enumerate(modelnicknames):
            err, _, diff_resp, diff_pred = get_pred_err(cid, batch, modelnames[nickname], part='probe',retur_diffs=True)
            ctx, prb, tme = err.shape
            del (err)
            if instances is None:
                diff_pred = diff_pred.reshape((-1, 100))
                if mm == 0:
                    diff_resp = diff_resp.reshape((-1, 100))

            else:
                ctx_pairs = [f'{c0:02}_{c1:02}' for c0, c1 in itt.combinations(range(ctx), 2)]
                view = instances.query(f"id == '{cid}'")
                # create an indexer to slice all significant instances in a single go
                ctx_idx = list()
                prb_idx = list()
                for rr, row in view.iterrows():
                    ctx_idx.append(ctx_pairs.index(row.context_pair))
                    prb_idx.append(row.probe - 1)

                diff_pred = diff_pred[ctx_idx, prb_idx, :]
                if mm == 0:
                    diff_resp = diff_resp[ctx_idx, prb_idx, :]

            # now that we have our selection, flip values so they are possitive and take means

            model_differences[mm,:] = np.where(diff_pred.sum(axis=-1,keepdims=True) < 0, diff_pred * -1, diff_pred).mean(axis=0)
            if mm == 0:
                response_differences = np.where(diff_resp.sum(axis=-1, keepdims=True) < 0,
                                                diff_resp * -1, diff_resp).mean(axis=0)

        label = cid

        return model_differences, response_differences, label

    # # parallelized method
    # output = jl.Parallel(n_jobs=4)(jl.delayed(inner)(cid) for cid in cellids)
    # pooled_diff_err, labels = zip(*output)
    # del output
    # pooled_diff_err = np.stack(pooled_diff_err, axis=0)

    # vanilla method
    pooled_mode_diff = list()
    pooled_resp_diff = list()
    labels = list()
    for cid in tqdm(cellids):
        model_differences, response_differences, label = inner(cid)
        pooled_mode_diff.append(model_differences)
        pooled_resp_diff.append(response_differences)
        labels.append(label)

    pooled_mode_diff = np.stack(pooled_mode_diff, axis=0)
    pooled_resp_diff = np.stack(pooled_resp_diff, axis=0)

    return pooled_mode_diff, pooled_resp_diff, labels




if __name__ == '__main__':
    from src.utils.subsets import cellid_fit_set
    config = ConfigParser()
    config.read_file(open(config_path / 'settings.ini'))

    # pooling delta errors
    recache = False
    cache_file = pl.Path(config['paths']['analysis_cache']) / '220406_labeled_pooled_errors'
    if cache_file.exists() and not recache:
        print(f'load zerroed errors from {cache_file}')
        # pooled_diff_err, labels = jl.load(cache_file)
    else:
        cellids = cellid_fit_set

        # load the real data to find the subset of neuron_context-pair_probes with significant modulation
        summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220310_ctx_mod_metric_DF_tstat_cluster_mass_BS'
        DF = jl.load(summary_DF_file)
        filtered = DF.query(
            f"metric == 'integral' and mult_comp_corr == 'bf_cp' and source == 'real' and "
            f"cluster_threshold == 0.05 and value > 0 and id in {list(cellids)}"
        )

        print(filtered)

        out = aggregate_all_errors(modelnicknames=['pop_lone_relu', 'pop_mod_relu'],
                                   floor='STRF_long_relu',
                                   cellids=cellids,
                                   batch=326,
                                   instances=filtered)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        jl.dump(out, cache_file)


    # pooling probe differences
    cache_file = pl.Path(config['paths']['analysis_cache']) / '220406_labeled_pooled_differences'
    recache = True
    if cache_file.exists() and not recache:
        print(f'load pooled differncese from {cache_file}')
        pooled_diff_err, labels = jl.load(cache_file)
    else:
        cellids = cellid_fit_set
        # cellids = {'TNC014a-22-2'}

        # load the real data to find the subset of neuron_context-pair_probes with significant modulation
        summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220310_ctx_mod_metric_DF_tstat_cluster_mass_BS'
        DF = jl.load(summary_DF_file)
        filtered = DF.query(
            f"metric == 'integral' and mult_comp_corr == 'bf_cp' and source == 'real' and "
            f"cluster_threshold == 0.05 and value > 0 and id in {list(cellids)}"
        )

        print(filtered)

        out = aggregate_all_differences(modelnicknames=['STRF_long_relu', 'pop_lone_relu', 'pop_mod_relu'],
                                   cellids=cellids,
                                   batch=326,
                                   instances=filtered)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        jl.dump(out, cache_file)
    pass




