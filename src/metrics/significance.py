import pathlib as pl
from configparser import ConfigParser
import numpy as np

def  _significance(array, mont_array, multiple_comparisons_axis, alpha=0.01, tails='both'):
    """
    calculates significance (boolean) for the values of array using the montecarlo method e.g. n simulations or shuffles of the
    original data in array. These n repetitions are specified in the mont_array, therefore mont_array should have the
    same shape as array plus an aditional first dimension for said repetitions. Correction for multiple comparisons is
    performed across the specified axies. alpha defines the threshold for considering pvalues as significant. tails
    specify wheter calculate a one tailed (upper or lower) or two tailed pvalue.
    :param array: ndarray e.g. Unit x Context x ...
    :param mont_array: ndarray e.g. Montecarlo x Unit x Context ...
    :param multiple_comparisons_axis: list of ints
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

    n_comparisons = np.prod(np.asarray(significance.shape)[np.asarray(multiple_comparisons_axis)])
    n_chance_comp = np.ceil(n_comparisons * alpha)

    # count the number of significant bins pooling acrooss the multiple comparisons axis, this is done independently
    # across all the other non specified axis, e.g. the Unit dimension
    sig_count = np.sum(significance, axis=tuple(multiple_comparisons_axis), keepdims=True)

    # creates a corrected significance by taking the significant bins of a groups of multiple comparisons, if the sum
    # of significant bins is over the chance comparisons threshold
    corrected_signif = np.where(sig_count > n_chance_comp, significance, np.full(significance.shape, False))

    # defines the confidence intervals as the top and bottom percentiles summing to alpha
    low = alpha * 100
    high = (1 - alpha) * 100
    confidence_interval = np.percentile(mont_array, [low, high], axis=0)

    return significance, corrected_signif, confidence_interval

