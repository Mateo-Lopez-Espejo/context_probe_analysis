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
    # since the last bin can be the first one (zeroth) temroally shift the times so the number reported is the next bin (first)
    t = label_dictionary['time']
    dt = np.mean(np.diff(t)) # delta time to add to every time point
    newshape = np.asarray(array.shape)
    newshape[-1] = 1
    ttile = np.ma.array(np.tile(t + dt, newshape), mask=array.mask)
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


##### Truncated or chunked metrics ######
def _mass_center_chunk_base(array, label_dictionary, start, end):
    """
    :array: shape Cell x Context_Pair x Probe x Time
    :label_dictionary: dict of lists describing each position in the array dimensions
    :start: start of slice in ms
    :end: end of slice in ms
    """
    t = np.asarray(label_dictionary['time'])
    idxr = np.argwhere((t >= start) & (t < end)).squeeze(axis=1)

    array = array[..., idxr]
    t = t[idxr]

    metric = np.sum(np.abs(array) * t[None, None, None, :], axis=3) / np.sum(np.abs(array), axis=3)
    metric = metric.filled(fill_value=0)
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


def _integral_chunk_base(array, label_dictionary, start, end, ignore_mask=False):
    """
    :array: shape Cell x Context_Pair x Probe x Time
    :label_dictionary: dict of lists describing each position in the array dimensions
    :start: start of slice in ms
    :end: end of slice in ms
    :ignore_mask: boolean, if true unmasks the data
    """
    t = np.asarray(label_dictionary['time'])
    idxr = np.argwhere((t >= start) & (t<end)).squeeze(axis=1)

    if ignore_mask:
        array = array.data

    array = array[..., idxr]
    t = t[idxr]

    metric = np.sum(np.abs(array), axis=3) * np.mean(np.diff(t))
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


def signif_abs_mass_center_truncated150(array, label_dictionary):
    return _mass_center_chunk_base(array, label_dictionary, start=150, end=1000)

def signif_abs_sum_trucated150(array, label_dictionary):
    return _integral_chunk_base(array, label_dictionary, start=150, end=1000)

def integral_A(array, label_dictionary):
    return _integral_chunk_base(array, label_dictionary, start=0, end=250)

def integral_B(array, label_dictionary):
    return _integral_chunk_base(array, label_dictionary, start=250, end=500)

def integral_C(array, label_dictionary):
    return _integral_chunk_base(array, label_dictionary, start=500, end=750)

def integral_D(array, label_dictionary):
    return _integral_chunk_base(array, label_dictionary, start=750, end=1000)


# calculates the integral across all data, independent of significance, see ignore_mask = True
def integral_nosig(array, label_dictionary):
    return _integral_chunk_base(array, label_dictionary, start=0, end=1000, ignore_mask=True)

def integral_nosig_A(array, label_dictionary):
    return _integral_chunk_base(array, label_dictionary, start=0, end=250, ignore_mask=True)

def integral_nosig_B(array, label_dictionary):
    return _integral_chunk_base(array, label_dictionary, start=250, end=500, ignore_mask=True)

def integral_nosig_C(array, label_dictionary):
    return _integral_chunk_base(array, label_dictionary, start=500, end=750, ignore_mask=True)

def integral_nosig_D(array, label_dictionary):
    return _integral_chunk_base(array, label_dictionary, start=750, end=1000, ignore_mask=True)


all_metrics = {'mass_center': signif_abs_mass_center,
               'integral': signif_abs_sum,
               'last_bin': signif_last_bin,
               'mass_center_trunc1.5': signif_abs_mass_center_truncated150,
               'integral_trunc1.5': signif_abs_sum_trucated150,
               'integral_A':integral_A,
               'integral_B':integral_B,
               'integral_C':integral_C,
               'integral_D':integral_D,
               'integral_nosig':integral_nosig,
               'integral_nosig_A':integral_nosig_A,
               'integral_nosig_B':integral_nosig_B,
               'integral_nosig_C':integral_nosig_C,
               'integral_nosig_D':integral_nosig_D,
               }


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
