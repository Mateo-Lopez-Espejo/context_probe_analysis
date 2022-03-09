import pathlib as pl
import warnings
from configparser import ConfigParser
import numpy as np
import numpy.ma as ma
import operator

def where_contiguous_chunks(array, axis, length, func='>=', individual_chunks=False):
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
    starts = np.argwhere(d == 1)
    stops = np.argwhere(d == -1)

    # counts the True chunks and select those with adecuate length
    ops = {'>': operator.gt, '<': operator.lt, '>=': operator.ge, '<=': operator.le, '==': operator.eq}
    good_chunks = ops[func](stops[:, axis] - starts[:, axis], length)


    if np.any(good_chunks):
        # iterates over the selected chunks and defines indices for the chunk dimension and the extra dimensions
        dim_idxs = list()
        for gc in np.argwhere(good_chunks).squeeze(axis=1):
            start = starts[gc, axis]
            stop = stops[gc, axis]

            chunk_all_dim = np.empty((stop - start, array.ndim), dtype=int)

            # make contiguous indices for the chunk dimension
            chunk_all_dim[:, axis] = np.arange(start, stop)

            if array.ndim > 1:
                # places the indices axis indices back in the right dimension position
                odidx = tuple(d for d in range(array.ndim) if d != axis)  # index into other dimension
                chunk_all_dim[:, odidx] = starts[gc, odidx]
                dim_idxs.append(chunk_all_dim)

    else:
        # a tuple of empty arrays can still be used as indices. the output of such slicing is an empty array too.
        dim_idxs = [np.empty(shape=(0,array.ndim), dtype=int)]

    # either returns a list of individual clusters or all clusters together.
    # numpy fancy indexing should be tuples of n arrays for n dimensions
    if individual_chunks:
        chunk_idx = [tuple(chunk.T) for chunk in dim_idxs]
    else:
        chunk_idx = tuple(np.concatenate(dim_idxs, axis=0).T)

    return chunk_idx


def _raw_pvalue(real_val, mont_array, tails='both'):
    mont_num = mont_array.shape[0]  # number of montecarlo repetitions
    if tails == 'both':
        pvalues = np.sum(np.abs(mont_array) >= np.abs(real_val), axis=0) / mont_num
        # pvalues = (np.sum(np.abs(mont_array) >= np.abs(real_val), axis=0)+1) / (mont_num+1)
    elif tails == 'greater':
        pvalues = np.sum((mont_array >= real_val), axis=0) / mont_num
    elif tails == 'lesser':
        pvalues = np.sum((mont_array >= real_val), axis=0) / mont_num
    else:
        raise ValueError("tails must be 'greater' 'lesser' or 'both'")

    return pvalues

def  _significance(pvalue, multiple_comparisons_axis=None, consecutive=0, alpha=0.01):
    """
    calculates significance (boolean) for the values of array using the montecarlo method e.g. n simulations or shuffles of the
    original data in array. These n repetitions are specified in the mont_array, therefore mont_array should have the
    same shape as array plus an aditional first dimension for said repetitions. Correction for multiple comparisons is
    performed across the specified axies. alpha defines the threshold for considering pvalues as significant.
    If consecutive is an integer and multiple_comparisons_axis is a singleton, counts consecutive True values instead
    of perfoming the multiple comparisons. Tails specify wheter calculate a one tailed (upper or lower) or two tailed
    pvalue.
    :param pvalue: ndarray e.g. Unit x Context x ...
    :param multiple_comparisons_axis: None, list of ints. default None.
        :param consecutive: int, default None. If int, multiple_comparisons_axis must be a singleton
    :param alpha: float [0:1]
    :param tails: str, 'lesser', 'Greater', 'Both', only used with full montecarlo array.
    :return:
    """
    # todo: rewrite and simplify so it only needs pvalues, some choice of multiple comparisons and an alpha (?)

    # chose what axis to consider for the correction, e.g only time, or time-ctx_pair-prb (best option)
    if multiple_comparisons_axis == None:
        n_comparisons = 1
    else:
        n_comparisons = np.prod(np.asarray(pvalue.shape)[np.asarray(multiple_comparisons_axis)])
        print(f'correcting for {n_comparisons} multiple comparisons, alpha: {alpha} -> {alpha/n_comparisons}')

    # using pvalue
    corrected_alpha = alpha/n_comparisons
    signif_corr = pvalue < corrected_alpha

    significance = signif_corr

    # sketchy consecutive criterium
    if consecutive > 0:
        if len(multiple_comparisons_axis) != 1:
            raise ValueError('when counting consecutive True, multiple_comparisons_axis must be singleton')

        print(f'considering contiguous chunks, overrides multiple comparisons')
        chunk_idx = where_contiguous_chunks(significance, multiple_comparisons_axis[0], consecutive, func='>=')
        chunk_signif = np.full_like(significance, False)
        chunk_signif[chunk_idx] = True
        significance = chunk_signif

    return significance

def get_clusters_mass(metric, threshold, axis, min_size=1, verbose=False):
    # defines threshold
    # find values with abs greater than threshold
    # does it in high and lows as the method requires clusters of the same sign
    high_vals = metric > threshold
    low_vals = metric < threshold*-1
    # find clusters, currently it can only look for cluster along a single axis
    high_chunks  = where_contiguous_chunks(high_vals, axis=axis, length=min_size, func='>=', individual_chunks=True)
    low_chunks  = where_contiguous_chunks(low_vals, axis=axis, length=min_size, func='>=', individual_chunks=True)

    # for each cluster calculates the sum of metric and stores in an array with the same shape as the original metric
    cluster_sum_arr = np.zeros_like(metric)
    for chunk in high_chunks + low_chunks:
        # calculate metric for clusters
        cluster_sum_arr[chunk]= metric[chunk].sum()

    if verbose:
        fig, axes= plt.subplots(1,2)
        eg_hi_idx = np.unravel_index(np.argmax(metric), metric.shape)
        eg_lo_idx = np.unravel_index(np.argmin(metric), metric.shape)

        for idx, ax in zip([eg_hi_idx, eg_lo_idx], axes):

            idx = np.s_[idx[0],idx[1],idx[2], :]

            ax.plot(metric[idx], label='metric')
            ax.plot(cluster_sum_arr[idx], label='cluster sum')
            ax.axhline(threshold, linestyle='--', color='black', label='threshold')
            ax.axhline(threshold*-1, linestyle='--', color='black')

        ax.legend()

        fig.show()

    return cluster_sum_arr


if __name__ == '__main__':
    import itertools as itt
    from src.metrics.consolidated_dprimes import single_cell_dprimes
    from src.utils.tools import shuffle_along_axis as shuffle
    import matplotlib.pyplot as plt
    from src.data.rasters import load_site_formated_raster
    from src.metrics.dprime import pairwise_dprimes

    meta = {'montecarlo': 1000,
            'raster_fs': 30,
            'reliability': 0.1,
            'smoothing_window': 0,
            'stim_type': 'permutations',
            'zscore': True}
    # site = 'TNC010a'
    site = 'ARM021b'
    dprime, pval_quantiles, goodcells, shuff_eg = single_cell_dprimes(site, contexts='all', probes='all', meta=meta)

    # check vanila bonferrony corrections and old chunk correction
    multiple_corrections = {'bf_cpt': ([1, 2, 3], 0),
                            'bf_ncpt': ([0, 1, 2, 3], 0),
                            'bf_cp': ([1, 2], 0),
                            'bf_t': ([3], 0),
                            'consecutive_3': ([3], 3)}

    for key, (mult, cont) in multiple_corrections.items():
        signif = _significance(pval_quantiles['pvalue'], mult, cont, alpha=0.05)

    # cluster finding fucntion
    cluster_arr = get_clusters_mass(dprime, 1, axis=-1, min_size=1, verbose=False)

