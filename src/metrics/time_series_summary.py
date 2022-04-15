import copy

import numpy as np
import pandas as pd

from src.utils import fits as fts
from src.utils.dataframes import ndim_array_to_long_DF as arr2df

"""
takes all the dprimes and pvalues, fits exponential decays to both the dprimes and the profiles of dprime
significance (derived from pvalues). 
This is done for all combinations of probe, and context transtion pairs.
This is done for single cells (SC), probewise dPCA (pdPCA) and full_dPCA (fdPCA).
"""


# defines all metrics functions with homogeneous input and output shapes
# start with significant absolute center of mass and significant absolute sum
# the input is always an array with shape Cell x Context_Pair x Probe x Time
# the output should be shape Cell x Context_pair x Probe (x Extra_values), an updated dict of axis labels
def signif_abs_sum(array, label_dictionary):
    t = label_dictionary['time']
    metric = np.sum(np.abs(array), axis=3) * np.mean(np.diff(t))
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


def signif_abs_mass_center(array, label_dictionary):
    t = label_dictionary['time']
    metric = np.sum(np.abs(array) * t[None, None, None, :], axis=3) / np.sum(np.abs(array), axis=3)
    metric = metric.filled(fill_value=0)
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


def signif_last_bin(array, label_dictionary):
    t = label_dictionary['time']
    newshape = np.asarray(array.shape)
    newshape[-1] = 1
    ttile = np.ma.array(np.tile(t, newshape), mask=array.mask)
    metric = np.max(ttile, axis=3)
    metric = metric.filled(fill_value=0)
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


def signif_abs_mean(array, label_dictionary):
    metric = np.mean(np.abs(array), axis=3)
    metric = metric.filled(fill_value=0)
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


##### Truncated metrics ######

def signif_abs_mass_center_truncated100(array, label_dictionary):
    # as the name implicates, truncate the metric calculation to values past the first 100 ms
    t = np.asarray(label_dictionary['time'])
    tidx = np.argwhere(t>100)[0,0]

    array = array[...,tidx:]
    t = t[tidx:]

    metric = np.sum(np.abs(array) * t[None, None, None, :], axis=3) / np.sum(np.abs(array), axis=3)
    metric = metric.filled(fill_value=0)
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


def signif_abs_sum_trucated100(array, label_dictionary):
    # as the name implicates, truncate the metric calculation to values past the first 100 ms
    t = np.asarray(label_dictionary['time'])
    tidx = np.argwhere(t > 100)[0,0]

    array = array[..., tidx:]
    t = t[tidx:]

    metric = np.sum(np.abs(array), axis=3) * np.mean(np.diff(t))
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


def signif_abs_mass_center_truncated150(array, label_dictionary):
    # as the name implicates, truncate the metric calculation to values past the first 100 ms
    t = np.asarray(label_dictionary['time'])
    tidx = np.argwhere(t > 150)[0,0]

    array = array[...,tidx:]
    t = t[tidx:]

    metric = np.sum(np.abs(array) * t[None, None, None, :], axis=3) / np.sum(np.abs(array), axis=3)
    metric = metric.filled(fill_value=0)
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


def signif_abs_sum_trucated150(array, label_dictionary):
    # as the name implicates, truncate the metric calculation to values past the first 100 ms
    t = np.asarray(label_dictionary['time'])
    tidx = np.argwhere(t > 150)[0,0]

    array = array[..., tidx:]
    t = t[tidx:]

    metric = np.sum(np.abs(array), axis=3) * np.mean(np.diff(t))
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


def signif_abs_mass_center_truncated200(array, label_dictionary):
    # as the name implicates, truncate the metric calculation to values past the first 100 ms
    t = np.asarray(label_dictionary['time'])
    tidx = np.argwhere(t > 200)[0,0]

    array = array[...,tidx:]
    t = t[tidx:]

    metric = np.sum(np.abs(array) * t[None, None, None, :], axis=3) / np.sum(np.abs(array), axis=3)
    metric = metric.filled(fill_value=0)
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


def signif_abs_sum_trucated200(array, label_dictionary):
    # as the name implicates, truncate the metric calculation to values past the first 100 ms
    t = np.asarray(label_dictionary['time'])
    tidx = np.argwhere(t > 200)[0,0]

    array = array[..., tidx:]
    t = t[tidx:]

    metric = np.sum(np.abs(array), axis=3) * np.mean(np.diff(t))
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


all_metrics = {'mass_center': signif_abs_mass_center,
               'integral': signif_abs_sum,
               'last_bin': signif_last_bin,
               'mass_center_trunc1': signif_abs_mass_center_truncated100,
               'integral_trunc1': signif_abs_sum_trucated100,
               'mass_center_trunc1.5': signif_abs_mass_center_truncated150,
               'integral_trunc1.5': signif_abs_sum_trucated150,
               'mass_center_trunc2': signif_abs_mass_center_truncated200,
               'integral_trunc2': signif_abs_sum_trucated200}


################################################

def metrics_to_DF(array, label_dictionary, metrics):
    metrics = {metric: all_metrics[metric] for metric in metrics}
    to_concat = list()
    for metric_name, func in metrics.items():
        metric_array, up_lab_dict = func(array, label_dictionary)
        df = arr2df(metric_array, up_lab_dict)
        df['metric'] = metric_name
        to_concat.append(df)

    site_DF = pd.concat(to_concat, axis=0, ignore_index=True)
    return site_DF
