import pathlib as pl
import warnings
from configparser import ConfigParser
from src.metrics.consolidated_metrics import _append_means_to_array
import numpy as np
import numpy.ma as ma
import operator

def where_contiguous_chunks(array, axis, length, func='>=', return_idx=False):
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

    # skips everything if there are not good chunks
    if not np.any(good_chunks):
        chunk_idx = tuple([np.empty(shape=(0,0), dtype=int) for dim in starts])
        return chunk_idx

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


def _raw_pvalue(real_val, mont_array, tails='both'):
    mont_num = mont_array.shape[0]  # number of montecarlo repetitions
    if tails == 'both':
        pvalues = np.sum(np.abs(mont_array) >= np.abs(real_val), axis=0) / mont_num
    elif tails == 'greater':
        pvalues = np.sum((mont_array >= real_val), axis=0) / mont_num
    elif tails == 'lesser':
        pvalues = np.sum((mont_array >= real_val), axis=0) / mont_num
    else:
        raise ValueError("tails must be 'greater' 'lesser' or 'both'")

    return pvalues

def _signif_quantiles(mont_array, alpha=(0.05, 0.01, 0.001)):
    """
    returns a dictionary of quantiles from the input montecarlo. These quantiles are effectively the thresholds
    for significance difference from the Montecarlo distribution on a two tailed test.
    :param mont_array: array with shape MontecarloReps x ...
    :param alpha: int, list. alpha or list of alpha values
    :return: dict of quatiles for each alpha
    """
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha]

    quantil_dict = dict()

    for alp in alpha:
        quantil_dict[alp] = np.quantile(mont_array, [alp/2, 1-alp/2], axis=0)

    return quantil_dict

def  _significance(array, mont_array, multiple_comparisons_axis=None, consecutive=0, alpha=0.01, verbose=False):
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
    :param tails: str, 'lesser', 'Greater', 'Both', only used with full montecarlo array.
    :return:
    """
    # todo: rewrite and simplify so it only needs pvalues, some choice of multiple comparisons and an alpha (?)

    # using quantiles deprecated method. difficult to run bonferroni on it.
    quantiles = mont_array[alpha]
    quant_signif = np.logical_or(array < quantiles[0, ...], quantiles[1, ...] < array)

    # chose what axis to consider for the correction, e.g only time, or time-ctx_pair-prb (best option)
    if multiple_comparisons_axis == None:
        n_comparisons = 1
    else:
        n_comparisons = np.prod(np.asarray(mont_array['pvalue'].shape)[np.asarray(multiple_comparisons_axis)])
        print(f'correcting for {n_comparisons} multiple comparisons, alpha: {alpha} -> {alpha/n_comparisons}')

    # using pvalue
    corrected_alpha = alpha/n_comparisons
    pval_signif = mont_array['pvalue'] < alpha
    pval_signif_corr = mont_array['pvalue'] < corrected_alpha

    significance = pval_signif_corr

    # sketchy consecutive criterium
    if consecutive > 0:
        if len(multiple_comparisons_axis) != 1:
            raise ValueError('when counting consecutive True, multiple_comparisons_axis must be singleton')

        print(f'considering contiguous chunks, overrides multiple comparisons')
        chunk_idx = where_contiguous_chunks(quant_signif, multiple_comparisons_axis[0], consecutive, func='>=')
        chunk_signif = np.full_like(quant_signif, False)
        chunk_signif[chunk_idx] = True
        significance = chunk_signif


    if verbose:
        bads = np.argwhere(pval_signif != quant_signif)

        bads_nt = np.unique(bads[:, :3], axis=0)
        for bad in bads_nt:
            print(bad)
            slice = np.s_[bad[0], bad[1], bad[2], :]

            plt.close('all')
            fig, ax = plt.subplots()
            ax.plot(array[slice], color='orange', label='dprime')
            ax.plot(quant_signif[slice], color='red', linestyle='dashed', label='quant_signif', alpha=0.5)
            # ax.plot(chunk_signif[slice], color='orange', linestyle='dotted', label='chunk_signif', alpha=0.5)
            ax.plot(pval_signif[slice], color='blue', linestyle='dashed', label='pval_signif', alpha=0.5)
            ax.plot(pval_signif_corr[slice], color='cyan', linestyle='dotted', label='pval_signif_corr', alpha=0.5)
            ax.plot(mont_array['pvalue'][slice], color='black', label='pvalue')

            ax.fill_between(np.arange(quantiles.shape[-1]),
                            quantiles[0, bad[0], bad[1], bad[2], :],
                            quantiles[1, bad[0], bad[1], bad[2], :], color='gray', alpha=0.1, label='cint')
            ax.axhline(alpha, linestyle=':', color='black')
            ax.axhline(corrected_alpha, linestyle=':', color='red')
            ax.legend()
            plt.show()

    return significance, mont_array[alpha]

def _mask_with_significance(dprime, significance, label_dictionary, mean_type='zeros', mean_signif_arr=None):
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

    if mean_type == 'zeros':
        # turns nonsigificant values into zeros and takes the means normally
        zeroed = np.where(significance, dprime, 0)
        dprime_means, mean_lable_dict = _append_means_to_array(zeroed, label_dictionary)
        masked_dprime_means = ma.array(dprime_means)

    elif mean_type == 'mean':
        # takes the mean of the significances and uses it to determine what mean values are significant
        dprime_means, mean_lable_dict = _append_means_to_array(dprime, label_dictionary)
        signif_means,_ = _append_means_to_array(significance, label_dictionary)
        masked_dprime_means = ma.array(dprime_means, mask=signif_means==0)

    elif mean_type == 'shuffles':
        # merges the imported  mean signif array into the dprime signifs
        dprime_means, mean_lable_dict = _append_means_to_array(dprime, label_dictionary)
        signif_means,_ = _append_means_to_array(significance, label_dictionary)
        signif_means[:,-1, :,:] = mean_signif_arr[:,-1, :,:]
        signif_means[:, :, -1,:] = mean_signif_arr[:, :, -1,:]
        masked_dprime_means = ma.array(dprime_means, mask=signif_means==0)

    else:
        raise ValueError(f'Unrecognized mean_type: {mean_type}')

    return masked_dprime_means, mean_lable_dict


def get_clusters_mass(metric, threshold):
    # defines threshold
    # find values with abs greater than threshold
    # does it in high and lows as the method requires clusters of the same sign
    high_vals = metric > threshold
    low_vals = metric > threshold
    # find clusters
    # hardcoding last axis, asumes its time.
    high_chunks  = where_contiguous_chunks(high_vals, axis=-1, length=2, func='>')
    low_chunks  = where_contiguous_chunks(low_vals, axis=-1, length=2, func='>')

    for chunk in high_chunks + low_chunks:
        # relate cluster postions to metric
        cluster_sum_arr = np.zeros_like(metric)
        # calculate metric for clusters
        cluster_sum_arr[chunk]= metric[chunk].sum()

    return cluster_sum_arr


if __name__ == '__main__':
    import itertools as itt
    from src.metrics.consolidated_dprimes import single_cell_dprimes
    from src.utils.tools import shuffle_along_axis as shuffle
    import matplotlib.pyplot as plt
    from src.data.rasters import load_site_formated_raster

    meta = {'alpha': 0.05,
            'montecarlo': 1000,
            'raster_fs': 30,
            'reliability': 0.1,
            'smoothing_window': 0,
            'stim_type': 'permutations',
            'zscore': True}
    site = 'TNC010a'
    dprime, pval_quantiles, goodcells, shuff_eg = single_cell_dprimes(site, contexts='all', probes='all', meta=meta)

    # check vanila bonferrony corrections and old chunk correction
    multiple_corrections = {'bf_cpt': ([1, 2, 3], 0),
                            'bf_ncpt': ([0, 1, 2, 3], 0),
                            'bf_t': ([3], 0),
                            'consecutive_3': ([3], 3)}

    for key, (mult, cont) in multiple_corrections.items():
        signif, quant = _significance(dprime, pval_quantiles, mult, cont, alpha=meta['alpha'])


    # chekc new cluster mass analsysi
    # trialR, goodcells = load_site_formated_raster(site, 'all', 'all', **meta)


    cluster_arr = get_clusters_mass(dprime, 1)

    print(cluster_arr.shape)
    print(cluster_arr[0,0,0,:])

    # ctx_pairs = list(itt.combinations(range(ctx), 2))
    # # shuffle of clusters
    # for cpn, (c0, c1) in enumerate(ctx_pairs):
    #     print(f"    context pair {c0:02d}_{c1:02d}")
    #     shuf_trialR = np.empty((meta['montecarlo'], rep, chn, 2, prb, tme))
    #     ctx_shuffle = trialR[:, :, (c0, c1), :, :].copy()  # trial, context, probe, time
    #
    #     for rr in range(meta['montecarlo']):
    #         shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)
    #
    #
