import itertools as itt
import pathlib as pl
from collections import defaultdict
from configparser import ConfigParser

import numpy as np
import scipy.stats as sst
from joblib import Memory

from src.data.rasters import load_site_formated_raster
from src.metrics.significance import _raw_pvalue, get_clusters_mass
from src.root_path import config_path
from src.utils.tools import shuffle_along_axis as shuffle

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

memory = Memory(str(pl.Path(config['paths']['analysis_cache']) / 'consolidated_tstat'))
# print(f'consolidated_tstat functions cache at:\n{memory.location}')


# @memory.cache
def single_cell_tstat_cluster_mass(site, contexts, probes, cluster_threshold, meta,
                                       load_fn=load_site_formated_raster):
    """
    calculated the T value between context for all probes and for all cells in a site.
    Defines significance of said tvalue using the cluster mass method (for time clusters), where a t value belongs to a
    cluster if it goes over the critical alpha value defined as cluster_threshold.
    The p value of the clusters themselves are defined using the permutation distributions of context id.
    Keeps one shuffled dprime, cluster and pvalue example.
    :param site: string  identifying the site
    :param probes: list of probe numbers
    :cluster_threshold: alpha value between 0 and 1, it is transformed internally to the 97.5 percentile of the T distribution
    :param meta: dict with meta parameters
    :return: dprime (ndarray with shape Unit x Ctx_pair x Probe x Time),
             pval_quantiles. dict of arrays of pval(same shape as dprime) or quantiles (upper-lower + dprime.shape)
             goocells (list of strings)
             shuffled_eg: dict with dprime and pvalue arrays for an example shuffled value (same shape as dprime)
    """
    # trialR, R, goodcells = load_site_formated_raster(site, contexts, probes, meta)
    trialR, goodcells = load_fn(site, contexts, probes, **meta)

    rep, chn, ctx, prb, tme = trialR.shape

    # shuffling on a context pair basis for consistancy with the pairwise dprime, and as a bonus, to deal with memory issues
    ctx_pairs = list(itt.combinations(range(ctx), 2))

    # pairwise mean difference, a simpler alternative to dprime
    ttest = np.empty((chn, len(ctx_pairs), prb, tme))
    for cpn, (c0, c1) in enumerate(ctx_pairs):
        ttest[:, cpn, :, :] = sst.ttest_ind(trialR[:, :, c0, :, :], trialR[:, :, c1, :, :], axis=0).statistic
    ttest = np.nan_to_num(ttest,copy=False, nan=0) # replaces nan to zero in place

    dprime = ttest

    # for the T values, considers the threshold an alpha level and finds the corresponding critical value in the
    # T distribution defined by the degrees of freedom of the two samples. i.e. n1-1 + n2-1
    crit_quatn = 1 - (cluster_threshold/2) # alpha divided by two since this is a two tailed test
    df = rep - 2
    cluster_threshold = sst.t.ppf(crit_quatn, df)

    clusters = get_clusters_mass(dprime, cluster_threshold, axis=-1)
    # shape Cell x CtxPair x Probe x Time

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle, then calculates the pairwise dprime

    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)

    # we need a single top quantil, and its a single value over all time points
    qnt_shape = list(dprime.shape)
    qnt_shape[-1] = 1
    quantiles = defaultdict(lambda: np.zeros(qnt_shape))
    pvalue = np.zeros_like(dprime)

    shuf_eg_dprime = np.zeros_like(dprime)
    shuf_eg_clust = np.zeros_like(dprime)
    shuf_eg_pvalue = np.zeros_like(dprime)

    for cpn, (c0, c1) in enumerate(ctx_pairs):
        print(f"    context pair {c0:02d}_{c1:02d}")
        shuf_trialR = np.empty((meta['montecarlo'], rep, chn, 2, prb, tme))
        ctx_shuffle = trialR[:, :, (c0, c1), :, :].copy()  # trial, context, probe, time

        for rr in range(meta['montecarlo']):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)

        # pairwise mean difference, a simpler alternative to dprime
        cpn_shuff_ttest = sst.ttest_ind(shuf_trialR[:, :, :, 0, :, :], shuf_trialR[:, :, :, 1, :, :], axis=1).statistic
        cpn_shuff_ttest = np.nan_to_num(cpn_shuff_ttest, copy=False, nan=0)
        cpn_shuff_dprime = np.expand_dims(cpn_shuff_ttest, axis=2)
        del (shuf_trialR)

        # for the shuffles, get only the value of the biggest cluster
        cpn_shuf_clstr = get_clusters_mass(cpn_shuff_dprime, cluster_threshold, axis=-1)
        cpn_shuf_clstr_max = np.max(np.abs(cpn_shuf_clstr), axis=-1, keepdims=True)

        # finally calculates pvalues for each individual cluster
        real_clstr = clusters[:, ctx_pairs.index((c0, c1)), ...][:, None, ...]
        cpn_pval = _raw_pvalue(real_clstr, cpn_shuf_clstr_max)
        pvalue[:, cpn, :, :] = cpn_pval.squeeze(axis=1)

        # saves raw quantiles for display alongside multiple comparisons corrections for:
        # context_pair*probe and neuron*context_pair*probe
        bf_corrections = dict(none=1,
                              bf_cp=len(ctx_pairs)*prb,
                              bf_ncp=chn*len(ctx_pairs)*prb)
        for corr_name, ncorr in bf_corrections.items(): #[0.05, 0.05 / 40, 0.05 / 550]:
            alpha = 0.05 / ncorr # asumes initial alpha of 0.05
            quantiles[corr_name][:, cpn, :, :] = np.quantile(cpn_shuf_clstr_max, 1 - alpha, axis=0).squeeze(axis=1)
        quantiles = {**quantiles}

        # saves onle last single random shuffle example and its corresponding pvalue
        sf_eg = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)
        # pairwise mean difference, a simpler alternative to dprime
        sf_eg_ttest = sst.ttest_ind(sf_eg[:, :, 0, :, :], sf_eg[:, :, 1, :, :], axis=0).statistic
        sf_eg_ttest = np.nan_to_num(sf_eg_ttest, copy=False, nan=0)
        sf_eg_dprime = np.expand_dims(sf_eg_ttest, axis=1)
        sf_eg_clust = get_clusters_mass(sf_eg_dprime, cluster_threshold, axis=-1)
        shuf_eg_dprime[:, cpn, :, :] = sf_eg_dprime.squeeze(axis=1)
        shuf_eg_clust[:, cpn, :, :] = sf_eg_clust.squeeze(axis=1)
        shuf_eg_pvalue[:, cpn, :, :] = _raw_pvalue(sf_eg_clust, cpn_shuf_clstr_max).squeeze(axis=1)
        del (cpn_shuff_dprime)

        if False and (ctx_pairs.index((0, 1)) == cpn):
            alpha = 0.05
            ncomp = prb * len(ctx_pairs)
            alpha_corr = alpha / ncomp

            from src.visualization.fancy_plots import squarefy
            eg_idx = np.s_[goodcells.index(cellid), ctx_pairs.index((0, 1)), 3 - 1, :]
            # eg_idx = np.s_[goodcells.index(cellid), 9, 3-1, :]

            if np.sum(dprime[eg_idx]) < 0:
                flip = -1
            else:
                flip = 1

            t = np.arange(30)
            d = dprime[eg_idx] * flip
            c = clusters[eg_idx] * flip
            s = cpn_shuf_clstr_max[:, eg_idx[0], 0, eg_idx[2], 0]
            p = pvalue[eg_idx]

            tt, dd = squarefy(t, d)
            _, cc = squarefy(t, c)
            tt, pp = squarefy(t, p)

            ci_c = quantiles[f'{alpha_corr:.5f}'][eg_idx]
            ci = quantiles[f'{alpha:.5f}'][eg_idx]

            fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=[8, 8])

            # raw data
            ax.hlines(s, 0, 30, color='gray', alpha=0.1)
            ax.plot(tt, dd, label='dprime')
            ax.plot(tt, cc, label='cluster')
            ax.axhline(ci_c, color='black', label='shuff_CI_corr')
            ax.axhline(cluster_threshold, color='red', linestyle=':', label='clust_threshold')

            ax.fill_between(tt, 0, 1, where=pp < alpha_corr,
                            color='green', alpha=0.5, transform=ax.get_xaxis_transform(), label='significant_corr')
            ax.legend()

            # pval, signif
            ax2.plot(tt, pp, label='pvalue', color='green')
            ax2.axhline(alpha_corr, color='brown', linestyle=':', label='alpha_bf')
            ax2.axhline(alpha, color='brown', linestyle='--', label='alpha')
            ax2.legend()

            # shuff dist and quantiles
            ax3.hist(s, bins=100, orientation='vertical')
            q0 = np.quantile(s, 1 - (alpha))
            q1 = np.quantile(s, 1 - (alpha_corr))

            ax3.axvline(ci_c, color='green', label='shuf_ci_corr')
            ax3.axvline(ci, color='red', linestyle='--', alpha=0.5, label='shuf_ci_raw')
            ax3.axvline(q0, color='blue', linestyle=':', alpha=0.5, label='shuf_ci_raw_recalc')
            ax3.axvline(q1, color='orange', linestyle=':', alpha=0.5, label='shuf_ci_corr_recalc')
            ax3.set_yscale('log')
            ax3.legend()

            fig.show()
            print(' ')

    # neat little output packages.
    clust_quant_pval = {'clusters': clusters, **quantiles, 'pvalue': pvalue, 't-threshold': cluster_threshold}
    shuffled_eg = dict(dprime=shuf_eg_dprime, clusters=shuf_eg_clust, pvalue=shuf_eg_pvalue)

    return dprime, clust_quant_pval, goodcells, shuffled_eg


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.visualization.fancy_plots import squarefy

    meta = {'reliability': 0.1,  # r value
            'smoothing_window': 0,  # ms
            'raster_fs': 30,
            'montecarlo': 1000,
            'zscore': True,
            'stim_type': 'permutations'}

    alpha = 0.05

    cellid, contexts, probes = 'ARM021b-36-8', (0, 1), 3  # huge difference between thresholds
    cellid, contexts, probes = 'CRD012b-13-1', (3, 4), 3  # huge difference between thresholds

    # out = single_cell_dprimes(site=id, contexts='all', probes='all',
    #                           meta=meta)

    cluster_threshold = 0.05
    # (dprime, clust_quant_pval, goodcells, shuffled_eg), _ = single_cell_tstat_cluster_mass.call(site=cellid,
    #                                                                                                 contexts='all',
    #                                                                                                 probes='all',
    #                                                                                                 cluster_threshold=cluster_threshold,
    #                                                                                                 meta=meta)
    dprime, clust_quant_pval, goodcells, shuffled_eg = single_cell_tstat_cluster_mass(site=cellid,
                                                                                      contexts='all',
                                                                                      probes='all',
                                                                                      cluster_threshold=cluster_threshold,
                                                                                      meta=meta)


    pvalue = clust_quant_pval['pvalue']
    clusters = clust_quant_pval['clusters']
    ct = clust_quant_pval['t-threshold']

    ncomp, fb_corr = dprime.shape[1] * dprime.shape[2], 'bf_cp'
    ncomp, fb_corr = dprime.shape[0] * dprime.shape[1] * dprime.shape[2], 'bf_ncp'

    alpha = 0.05
    alpha_corr = alpha / ncomp

    if dprime.shape[1] == 10:
        ctx = 5
    elif dprime.shape[1] == 55:
        ctx = 11
    else:
        raise ValueError('unknown nuber of context pairs')

    ctx_pairs = list(itt.combinations(range(ctx), 2))

    eg_idx = np.s_[goodcells.index(cellid), ctx_pairs.index(contexts), probes - 1, :]

    if np.sum(dprime[eg_idx]) < 0:
        flip = -1
    else:
        flip = 1

    t = np.arange(30)
    d = dprime[eg_idx] * flip
    c = clusters[eg_idx] * flip
    p = pvalue[eg_idx]

    tt, dd = squarefy(t, d)
    _, cc = squarefy(t, c)
    tt, pp = squarefy(t, p)

    ci_c = clust_quant_pval[fb_corr][eg_idx]
    ci = clust_quant_pval[fb_corr][eg_idx]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=[8, 8])

    # raw data
    ax.plot(tt, dd, label='dprime')
    ax.plot(tt, cc, label='cluster')
    ax.axhline(ci_c, color='black', label='shuff_CI_corr')
    ax.axhline(ct, color='red', linestyle=':', label='clust_threshold')

    ax.fill_between(tt, 0, 1, where=pp < alpha_corr,
                    color='green', alpha=0.5, transform=ax.get_xaxis_transform(), label='significant_corr')
    ax.legend()

    # pval, signif
    ax2.plot(tt, pp, label='pvalue', color='green')
    ax2.axhline(alpha_corr, color='brown', linestyle=':', label='alpha_bf')
    ax2.axhline(alpha, color='brown', linestyle='--', label='alpha')
    ax2.legend()

    fig.show()
