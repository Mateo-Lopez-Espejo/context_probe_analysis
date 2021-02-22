import pathlib as pl
from configparser import ConfigParser
from src.metrics.consolidated_metrics import _append_means_to_array
import numpy as np
import numpy.ma as ma

def  _significance(array, mont_array, multiple_comparisons_axis, alpha=0.01, tails='both'):
    """
    calculates significance (boolean) for the values of array using the montecarlo method e.g. n simulations or shuffles of the
    original data in array. These n repetitions are specified in the mont_array, therefore mont_array should have the
    same shape as array plus an aditional first dimension for said repetitions. Correction for multiple comparisons is
    performed across the specified axies. alpha defines the threshold for considering pvalues as significant. tails
    specify wheter calculate a one tailed (upper or lower) or two tailed pvalue.
    :param array: ndarray e.g. Unit x Context x ...
    :param mont_array: ndarray e.g. Montecarlo x Unit x Context ...
    :param multiple_comparisons_axis: list of ints, None
    :param alpha: float [0:1]
    :param tails: str, 'lesser', 'Greater', 'Both'
    :return:
    """

    mont_num = mont_array.shape[0] # number of montecarlo repetitions

    if tails is 'both':
        pvalues = np.sum(np.abs(mont_array) >= np.abs(array), axis=0) / mont_num
    elif tails == 'greater':
        pvalues = np.sum((mont_array >= array), axis=0) / mont_num
    elif tails == 'lesser':
        pvalues = np.sum((mont_array >= array), axis=0) / mont_num
    else:
        raise ValueError("tails must be 'greater' 'lesser' or 'both'")

    significance = pvalues <= alpha

    if multiple_comparisons_axis is not None:

        n_comparisons = np.prod(np.asarray(significance.shape)[np.asarray(multiple_comparisons_axis)])
        n_chance_comp = np.ceil(n_comparisons * alpha)

        # count the number of significant bins pooling acrooss the multiple comparisons axis, this is done independently
        # across all the other non specified axis, e.g. the Unit dimension
        sig_count = np.sum(significance, axis=tuple(multiple_comparisons_axis), keepdims=True)

        # creates a corrected significance by taking the significant bins of a groups of multiple comparisons, if the sum
        # of significant bins is over the chance comparisons threshold
        significance = np.where(sig_count > n_chance_comp, significance, np.full(significance.shape, False))

    # defines the confidence intervals as the top and bottom percentiles summing to alpha
    low = alpha * 100
    high = (1 - alpha) * 100
    confidence_interval = np.percentile(mont_array, [low, high], axis=0)

    return significance, confidence_interval


def _mask_with_significance(dprime, significance, label_dictionary, mean_type='zeros'):
    """
    uses the significance array to mask the dprime for later analysis. The critial use of this fuction is to deal with
    different aproaches on how to deal with the significance for the mean across context_pairs, probes or both.
    zeros:
    :param dprime: arraye of dprime values
    :param significance: boolean array of the significance of said dprimes
    :param label_dict: dictionary of the dimension labels for dprime
    :param mean_type: how to deal with the sigificance for the mean values
    :return: masked array containing appended mean values, ready for metric calculation
    """

    if mean_type == "zeros":
        # turns nonsigificant values into zeros and takes the means normally
        zeroed = np.where(significance, dprime, 0)
        dprime_means, mean_lable_dict = _append_means_to_array(zeroed, label_dictionary)
        masked_dprime_means = ma.array(dprime_means)

    elif mean_type == "mean":
        # takes the mean of the significances and uses it to determine what mean values are significant
        dprime_means, mean_lable_dict = _append_means_to_array(dprime, label_dictionary)
        signif_means,_ = _append_means_to_array(significance, label_dictionary)
        masked_dprime_means = ma.array(dprime_means, mask=signif_means==0)
    else:
        raise ValueError(f'Unrecognized mean_type: {mean_type}')

    return masked_dprime_means, mean_lable_dict

