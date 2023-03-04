import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import numpy as np
import scipy.stats as sst
from joblib import Memory
from tqdm import tqdm

from src.data.tensor_loaders import tensor_loaders
from src.metrics.significance import _raw_pvalue, get_clusters_mass
from src.root_path import config_path
from src.utils.tools import shuffle_along_axis as shuffle

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

memory = Memory(str(pl.Path(config['paths']['analysis_cache']) / 'consolidated_tstat'))


@memory.cache
def tstat_cluster_mass(site, cluster_threshold, montecarlo, raster_meta, load_fn):
    """
    calculated the T value between context for all probes and for all channels in a site.
    Defines significance of said tvalue using the cluster mass method (for time clusters), where a t value belongs to a
    cluster if it goes over the critical alpha value defined as cluster_threshold.
    The p value of the clusters themselves are defined using the permutation distributions of context id.
    Keeps one shuffled tstats, cluster and pvalue example.
    :param site: string  identifying the site
    :param probes: list of probe numbers
    :cluster_threshold: alpha value between 0 and 1, it is transformed internally to the 97.5 percentile of the T distribution
    :param meta: dict with meta parameters
    :load_fn: str, specifiese what function
    :return: tstats (ndarray with shape Chan x Ctx_pair x Probe x Time),
             pval_quantiles. dict of arrays of pval(same shape as tstats) or quantiles (upper-lower + tstats.shape)
             goocells (list of strings)
             shuffled_eg: dict with tstats and pvalue arrays for an example shuffled value (same shape as tstats)
    """
    trialR, goodcells = tensor_loaders[load_fn](site, recache_rec=True, **raster_meta)

    rep, chn, ctx, prb, tme = trialR.shape
    ctx_pairs = list(itt.combinations(range(ctx), 2))

    # vanila T value as difference statistic
    tstats = np.empty((chn, len(ctx_pairs), prb, tme))
    for cpidx, (c0, c1) in enumerate(ctx_pairs):
        tstats[:, cpidx, :, :] = sst.ttest_ind(trialR[:, :, c0, :, :], trialR[:, :, c1, :, :], axis=0).statistic
    tstats = np.nan_to_num(tstats, copy=False, nan=0)  # replaces nan to zero in place

    # for the T values, considers the threshold an alpha level and finds the corresponding critical value in the
    # T distribution defined by the degrees of freedom of the two samples. i.e. n1-1 + n2-1
    crit_quatn = 1 - (cluster_threshold / 2)  # alpha divided by two since this is a two tailed test
    df = rep * 2 - 2  # degrees of freedom, both samples have the same number of observation i.e. repetitions
    cluster_threshold = sst.t.ppf(crit_quatn, df)

    clusters = get_clusters_mass(tstats, cluster_threshold, axis=-1)

    # Shuffles the rasters n times and organizes in an array with the same shape as the source raster plus a prepended
    # dimension with size n containing each shuffle, then calculates the pairwise tstats

    print(f"\nshuffling {montecarlo} times")
    rng = np.random.default_rng(42)

    # we need a single top quantile, and it s a single value over all time points
    # context_pair*probe and neuron*context_pair*probe
    bf_corrections = dict(none=1,
                          bf_cp=len(ctx_pairs) * prb,
                          bf_ncp=chn * len(ctx_pairs) * prb)
    qnt_shape = list(tstats.shape)
    qnt_shape[-1] = 1
    quantiles = {corr_name: np.zeros(qnt_shape) for corr_name in bf_corrections.keys()}
    pvalue = np.zeros_like(tstats)

    shuf_eg_tstats = np.zeros_like(tstats)
    shuf_eg_clust = np.zeros_like(tstats)
    shuf_eg_pvalue = np.zeros_like(tstats)

    for (nidx, cellid), (cpidx, (c0, c1)) in tqdm(
            itt.product(enumerate(goodcells), enumerate(ctx_pairs)),
            total=len(goodcells) * len(ctx_pairs)):
        # print(f"    {cellid}: context pair {c0:02d}_{c1:02d}")
        shuf_trialR = np.empty((montecarlo, rep, 2, prb, tme))
        ctx_shuffle = trialR[:, nidx, (c0, c1), :, :].copy()  # trial, context, time

        for rr in range(montecarlo):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=1, indie_axis=0, rng=rng)

        # pairwise mean difference, a simpler alternative to tstats
        cpn_shuff_ttest = sst.ttest_ind(shuf_trialR[:, :, 0, :, :], shuf_trialR[:, :, 1, :, :], axis=1).statistic
        cpn_shuff_ttest = np.nan_to_num(cpn_shuff_ttest, copy=False, nan=0)
        del (shuf_trialR)

        # for the shuffles, get only the value of the biggest cluster regardless of sign
        cpn_shuf_clstr = get_clusters_mass(cpn_shuff_ttest, cluster_threshold, axis=-1)
        cpn_shuf_clstr_max = np.max(np.abs(cpn_shuf_clstr), axis=-1, keepdims=True)

        # finally, calculates p values for each individual cluster
        real_clstr = clusters[nidx, cpidx, :, :]
        cpn_pval = _raw_pvalue(real_clstr, cpn_shuf_clstr_max)
        pvalue[nidx, cpidx, :, :] = cpn_pval

        # saves raw quantiles for display alongside multiple comparisons corrections
        for corr_name, ncorr in bf_corrections.items():  # [0.05, 0.05 / 40, 0.05 / 550]:
            alpha = 0.05 / ncorr  # asumes initial alpha of 0.05
            quantiles[corr_name][nidx, cpidx, :, :] = np.quantile(cpn_shuf_clstr_max, 1 - alpha, axis=0)

        # saves onle last single random shuffle example and its corresponding pvalue
        sf_eg = shuffle(ctx_shuffle, shuffle_axis=1, indie_axis=0, rng=rng)
        sf_eg_tstat = sst.ttest_ind(sf_eg[:, 0, :, :], sf_eg[:, 1, :, :], axis=0).statistic
        sf_eg_tstat = np.nan_to_num(sf_eg_tstat, copy=False, nan=0)
        sf_eg_clust = get_clusters_mass(sf_eg_tstat, cluster_threshold, axis=-1)
        shuf_eg_tstats[nidx, cpidx, :, :] = sf_eg_tstat
        shuf_eg_clust[nidx, cpidx, :, :] = sf_eg_clust
        shuf_eg_pvalue[nidx, cpidx, :, :] = _raw_pvalue(sf_eg_clust, cpn_shuf_clstr_max)
        del (cpn_shuff_ttest)

    # neat little output packages.
    clust_quant_pval = {'clusters': clusters, **quantiles, 'pvalue': pvalue, 't-threshold': cluster_threshold}
    shuffled_eg = dict(dprime=shuf_eg_tstats, clusters=shuf_eg_clust, pvalue=shuf_eg_pvalue)

    return tstats, clust_quant_pval, goodcells, shuffled_eg



if __name__ == "__main__":
    def test():
        import matplotlib.pyplot as plt
        from src.visualization.utils import squarefy

        raster_meta = {'reliability': 0.1,  # r value
                        'smoothing_window': 0,  # ms
                        'raster_fs': 20,
                        'zscore': True,
                        'stim_type': 'permutations'}

        cellid, contexts, probes = 'CRD012b-13-1', (3, 4), 3  # huge difference between thresholds
        cellid, contexts, probe = 'TNC024a-27-2', (1, 10), 3 # great example of PEG data
        siteid = cellid[:7]
        cluster_threshold = 0.05
        #
        # (tstats, clust_quant_pval, goodcells, shuffled_eg), _ = tstat_cluster_mass.call(
        #     site=siteid,cluster_threshold=cluster_threshold,montecarlo=10,raster_meta=raster_meta,load_fn='SC'
        # )
        #
        tstats, clust_quant_pval, goodcells, shuffled_eg = tstat_cluster_mass(
            site=siteid,cluster_threshold=cluster_threshold,montecarlo=10,raster_meta=raster_meta,load_fn='SC'
        )


        pvalue = clust_quant_pval['pvalue']
        clusters = clust_quant_pval['clusters']
        ct = clust_quant_pval['t-threshold']

        ncomp, fb_corr = 1, 'none'
        # ncomp, fb_corr = tstats.shape[1] * tstats.shape[2], 'bf_cp'
        # ncomp, fb_corr = tstats.shape[0] * tstats.shape[1] * tstats.shape[2], 'bf_ncp'

        alpha = 0.05
        alpha_corr = alpha / ncomp

        if tstats.shape[1] == 10:
            ctx = 5
        elif tstats.shape[1] == 55:
            ctx = 11
        else:
            raise ValueError('unknown nuber of context pairs')

        ctx_pairs = list(itt.combinations(range(ctx), 2))

        # eg_idx = np.s_[goodcells.index(cellid), ctx_pairs.index(contexts), probes - 1, :]
        eg_idx = np.s_[1, ctx_pairs.index(contexts), probes - 1, :]

        if np.sum(tstats[eg_idx]) < 0:
            flip = -1
        else:
            flip = 1

        t = np.arange(raster_meta['raster_fs'])
        d = tstats[eg_idx] * flip
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
    test()
