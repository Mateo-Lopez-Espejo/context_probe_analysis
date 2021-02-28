import pathlib as pl
from configparser import ConfigParser
from src.metrics.consolidated_metrics import _append_means_to_array
import numpy as np
import numpy.ma as ma
import operator

def where_contiguous_chunks(array, axis, length, func='>='):
    """
    find the indices where a contiguous number of True values along axis are. finds True chunks equal or longer than the
    specified number
    :param array: boolean array
    :param axis: int, axis over which to count for contiguous True
    :param length: chunks of length equal or greater are selected
    :func: the comparison between the chunk lengths and desired length
    :return: tuple of indices
    """

    # sanitizes input
    if axis == -1: axis = array.ndim-1

    # find the start and end of True chunks
    d = np.diff(array.astype(int), axis=axis, prepend=0, append=0)
    starts = np.where(d == 1)
    stops = np.where(d == -1)

    # counts the True chunks and select those with adecuate length
    ops = {'>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le, '==': operator.eq}
    good_chunks = ops[func](stops[axis] - starts[axis], length)

    # save the indices from the other dimensions
    if array.ndim > 1:
        other_dims_idx = list(starts)
        other_dims_idx.pop(axis)
        other_dims_idx =  np.asarray(other_dims_idx)

    # iterates over the selected chunks and defines indices for the chunk dimension and the extra dimensions
    chunk_dim_idx = list()
    extra_dim_idx = list()
    for gc in np.where(good_chunks)[0]:
        start = starts[axis][gc]
        stop = stops[axis][gc]

        # make contiguous indices for the chunk dimension
        chunk_dim_idx.append(np.arange(start, stop))

        # repeat chunk length times the indices for the other dimensions
        if array.ndim > 1:
            extra_dim_idx.append(np.repeat(other_dims_idx[:,[gc]], repeats=stop-start, axis=1))

    chunk_dim_idx = np.concatenate(chunk_dim_idx, axis=0)

    if array.ndim > 1:
        extra_dim_idx = np.concatenate(extra_dim_idx, axis=1)

        # places the chunk dimension indices alongside the other dimensions in a tuple in the original position
        chunk_idx = list(extra_dim_idx.__iter__())
        chunk_idx.insert(axis, chunk_dim_idx)
        chunk_idx = tuple(chunk_idx)
    else:
        chunk_idx = (chunk_dim_idx,)

    return chunk_idx

def  _significance(array, mont_array, multiple_comparisons_axis=None, consecutive=None, alpha=0.01, tails='both'):
    """
    calculates significance (boolean) for the values of array using the montecarlo method e.g. n simulations or shuffles of the
    original data in array. These n repetitions are specified in the mont_array, therefore mont_array should have the
    same shape as array plus an aditional first dimension for said repetitions. Correction for multiple comparisons is
    performed across the specified axies. alpha defines the threshold for considering pvalues as significant.
    If consecutive is an integer and multiple_comparisons_axis is a singleton, counts consecutive True values instead
    of perfoming the multiple comparisons. Tails specify wheter calculate a one tailed (upper or lower) or two tailed
    pvalue.
    :param array: ndarray e.g. Unit x Context x ...
    :param mont_array: ndarray e.g. Montecarlo x Unit x Context ...
    :param multiple_comparisons_axis: None, list of ints. default None.
        :param consecutive: int, default None. If int, multiple_comparisons_axis must be a singleton
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

    # does corrections, otherwise passe the raw significance
    if multiple_comparisons_axis is not None:
        # counts consecutive True values
        if isinstance(consecutive, int):
            if len(multiple_comparisons_axis) != 1:
                raise ValueError('when counting consecutive True, multiple_comparisons_axis must be singleton')

            chunk_idx = where_contiguous_chunks(significance, multiple_comparisons_axis[0], consecutive, func='>=')
            chunk_signif = np.full_like(significance)
            chunk_signif[chunk_idx] = True
            significance = chunk_signif

        # corrects for multiple comparisons
        elif consecutive is None:

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

