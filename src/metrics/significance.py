import pathlib as pl
import warnings
from configparser import ConfigParser
from src.metrics.consolidated_metrics import _append_means_to_array
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
        # pvalues = np.sum(np.abs(mont_array) >= np.abs(real_val), axis=0) / mont_num
        pvalues = (np.sum(np.abs(mont_array) >= np.abs(real_val), axis=0)+1) / (mont_num+1)
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
        # using quantiles deprecated method. difficult to run bonferroni on it.
        quantiles = mont_array[alpha]
        quant_signif = np.logical_or(array < quantiles[0, ...], quantiles[1, ...] < array)

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

    return significance

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

    meta = {'alpha': 0.05,
            'montecarlo': 1000,
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
                            'bf_t': ([3], 0),
                            'consecutive_3': ([3], 3)}

    for key, (mult, cont) in multiple_corrections.items():
        signif, quant = _significance(dprime, pval_quantiles, mult, cont, alpha=meta['alpha'], verbose=False)


    # cluster finding fucntion
    cluster_arr = get_clusters_mass(dprime, 1, axis=-1, min_size=1, verbose=False)


    # # clustering algorithm with shuffle test
    # debug = True
    # trialR, goodcells = load_site_formated_raster(site, 'all', 'all', **meta)
    # rep, chn, ctx, prb, tme = trialR.shape
    # rng = np.random.default_rng(42)
    #
    # dprime = pairwise_dprimes(trialR, observation_axis=0, condition_axis=2)
    # threshold = 1
    # clusters = get_clusters_mass(dprime, threshold, axis=-1)
    #
    # ctx_pairs = list(itt.combinations(range(ctx), 2))
    # # shuffle of clusters
    # montecarlo = 100
    # for cpn, (c0, c1) in enumerate(ctx_pairs):
    #     print(f"    context pair {c0:02d}_{c1:02d}")
    #     shuf_trialR = np.empty((montecarlo, rep, chn, 2, prb, tme))
    #     ctx_shuffle = trialR[:, :, (c0, c1), :, :].copy()  # trial, context, probe, time
    #
    #     for rr in range(montecarlo):
    #         shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)
    #
    #     ctp_shuff_dprime = pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=3)
    #     del (shuf_trialR)
    #
    #     # for the shuffles, get only the value of the biggest cluster
    #     cpn_shuf_clstr_max = np.max(np.abs(get_clusters_mass(ctp_shuff_dprime, threshold, axis=-1)), axis=-1)
    #     cpn_shuf_clstr_max = np.expand_dims(cpn_shuf_clstr_max, axis=-1)
    #
    #     # calculates pvalus for each cluster based on the permutation biggest cluster distribution
    #
    #     real_clstr = clusters[:, ctx_pairs.index((c0, c1)), ...][:, None, ...]
    #     pvalue = _raw_pvalue(real_clstr, cpn_shuf_clstr_max)
    #
    #
    #     # example plots for debuggin
    #     if debug:
    #         eg_idx = np.unravel_index(np.argmax(np.absolute(real_clstr)), shape=real_clstr.shape)
    #         eg_idx = np.unravel_index(np.argmax(real_clstr*-1), shape=real_clstr.shape)
    #         eg_idx = np.s_[eg_idx[0], eg_idx[1], eg_idx[2],:]
    #
    #         fig, ax = plt.subplots(figsize=[8,8])
    #         ax.plot(dprime[eg_idx[0],ctx_pairs.index((c0, c1)),eg_idx[2],:], label='dprime')
    #         max_clust = np.max(np.absolute(real_clstr[eg_idx]))
    #         norm_clust = real_clstr[eg_idx] / max_clust
    #         ax.plot(norm_clust, label='cluster')
    #         ax.plot(pvalue[eg_idx], label='pvalue')
    #         norm_shuf = cpn_shuf_clstr_max[(np.s_[:], )+eg_idx].squeeze() / max_clust
    #         ax.hlines(norm_shuf,0,30, alpha=0.1, color='gray', label='shuf_max_clust')
    #         ax.hlines(norm_shuf*-1,0,30, alpha=0.1, color='gray')
    #         ax.axhline(threshold, color='red', linestyle=':', label='clust_threshold')
    #         ax.axhline(threshold*-1, color='red', linestyle=':')
    #         ax.axhline(meta['alpha'], color='brown', linestyle='--', label='alpha')
    #         ax.legend()
    #         fig.show()
