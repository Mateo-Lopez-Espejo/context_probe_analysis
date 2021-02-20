import copy
import numpy as np
import pandas as pd
from src.utils.tools import ndim_array_to_long_DF as arr2df

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
    metric = np.sum(np.abs(array) * t[None,None,None,:], axis=3) / np.sum(np.abs(array), axis=3)
    updated_label_dict = copy.deepcopy(label_dictionary)
    _ = updated_label_dict.pop('time')
    return metric, updated_label_dict


all_metrics = {'significant_abs_mass_center': signif_abs_mass_center,
               'significant_abs_sum': signif_abs_sum}

################################################
def _add_means_to_array(array, label_dictionary):
    """
    defines means across probes, context_pairse or both, and append to a copy of the original array. Updates the dimension
    label dictionary
    :param array: array of shape Cell x Context_Pair x Probe x Time
    :param label_dictionary: dimension labels dictionary
    :return: array, dictionary
    """
    newshape = np.asarray(array.shape) + [0,1,1,0] # add space for the Context_Pair, and/or Probe
    array_with_means = np.full(newshape, np.nan)

    array_with_means[:, :-1, :-1, :] = array #original values
    array_with_means[:, -1, :-1, :] = np.mean(array, axis=1) # context_pair mean
    array_with_means[:, :-1, -1, :] = np.mean(array, axis=2) # probe mean
    array_with_means[:, -1, -1, :] = np.mean(array, axis=(1,2)) # full mean

    # updates dimension label dictionary
    new_lab_dict = copy.deepcopy(label_dictionary)
    new_lab_dict['context_pair'].append('mean')
    new_lab_dict['probe'].append('mean')

    return array_with_means, new_lab_dict


def metrics_to_DF(array, label_dictionary, metrics):
    metrics = {metric: all_metrics[metric] for metric in metrics}

    means, means_lab_dict = _add_means_to_array(array, label_dictionary)
    site_DF = pd.DataFrame()
    for metric_name, func in metrics.items():
        sig_abs_sum, sas_lab = signif_abs_sum(means, means_lab_dict)
        df = arr2df(sig_abs_sum, sas_lab)
        df['metric'] = metric_name
        site_DF = site_DF.append(df)

    return site_DF