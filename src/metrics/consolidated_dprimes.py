import itertools as itt
import pathlib as pl
from collections import defaultdict
from configparser import ConfigParser

import numpy as np
from joblib import Memory

from src.data import LDA as cLDA, dPCA as cdPCA
from src.data.rasters import load_site_formated_raster
from src.metrics import dprime as cDP
from src.metrics.significance import _raw_pvalue, get_clusters_mass
from src.root_path import config_path
from src.utils.tools import shuffle_along_axis as shuffle

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

memory = Memory(str(pl.Path(config['paths']['analysis_cache']) / 'consolidated_dprimes'))


# these functionse should operate over site, and a probe, and return a pairwise dprime between contexts, plus the shuffled
# and simulated montecarlos and  good cells
@memory.cache
def single_cell_dprimes(site, contexts, probes, meta, load_fn=load_site_formated_raster):
    """
    calculated the dprime between context for all probes and for all cells in a site. Calculates dprime using
    montecarlo shuffling of the context identity, also keeps one shuffled example for other statistical analysis
    :param site: string  identifying the site
    :param probes: list of probe numbers
    :param meta: dict with meta parameters
    :return: dprime (ndarray with shape Unit x Ctx_pair x Probe x Time),
             pval_quantiles. dict of arrays of pval(same shape as dprime) or quantiles (upper-lower + dprime.shape)
             goocells (list of strings)
             shuffled_eg: dict with dprime and pvalue arrays for an example shuffled value (same shape as dprime)
    """
    # trialR, R, goodcells = load_site_formated_raster(site, contexts, probes, meta)
    trialR, goodcells = load_fn(site, contexts, probes, **meta)

    rep, chn, ctx, prb, tme = trialR.shape

    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, condition_axis=2)  # shape Cell x CtxPair x Probe x Time

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle, then calculates the pairwise dprime

    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)

    # shuffling on a context pair basis for consistancy with the pairwise dprime, and as a bonus, to deal with memory issues
    ctx_pairs = list(itt.combinations(range(ctx), 2))

    qnt_shape = (2,) + dprime.shape
    quantiles = defaultdict(lambda: np.zeros(qnt_shape))
    pvalue = np.zeros_like(dprime)

    shuf_eg_dprime = np.zeros_like(dprime)
    shuf_eg_pvalue = np.zeros_like(dprime)

    for cpn, (c0, c1) in enumerate(ctx_pairs):
        print(f"    context pair {c0:02d}_{c1:02d}")
        shuf_trialR = np.empty((meta['montecarlo'], rep, chn, 2, prb, tme))
        ctx_shuffle = trialR[:, :, (c0, c1), :, :].copy()  # trial, context, probe, time

        for rr in range(meta['montecarlo']):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)

        neur_shuff_dprime = cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=3)
        del (shuf_trialR)

        # saves neuron pval for refined multiple comparisons with flexible alphas
        real_dprime = dprime[:, ctx_pairs.index((c0, c1)), ...][:, None, ...]
        pvalue[:, cpn, :, :] = _raw_pvalue(real_dprime, neur_shuff_dprime).squeeze(axis=1)

        # # saves quantiles mostly for display, i.e. gray confidense intervale on dprime plots
        # neur_shuff_quantils = _signif_quantiles(neur_shuff_dprime)
        # for alpha, qntls in neur_shuff_quantils.items():
        #     quantiles[alpha][:, :, cpn, :, :] = qntls.squeeze(axis=2)

        # saves onle last single random shuffle example and its corresponding pvlaue
        sf_eg = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)
        sf_eg_dprime = cDP.pairwise_dprimes(sf_eg, observation_axis=0, condition_axis=2)
        shuf_eg_dprime[:, cpn, :, :] = sf_eg_dprime.squeeze(axis=1)
        shuf_eg_pvalue[:, cpn, :, :] = _raw_pvalue(sf_eg_dprime, neur_shuff_dprime).squeeze(axis=1)
        del (neur_shuff_dprime)

    # neat little output packages.
    pval_quantiles = {'pvalue': pvalue, **quantiles}
    shuffled_eg = dict(dprime=shuf_eg_dprime, pvalue=shuf_eg_pvalue)

    return dprime, pval_quantiles, goodcells, shuffled_eg


@memory.cache
def single_cell_dprimes_cluster_mass(site, contexts, probes, cluster_threshold, meta,
                                     load_fn=load_site_formated_raster):
    """
    calculated the dprime between context for all probes and for all cells in a site.
    Defines significance of said dprime using the cluster mass method (for time clusters) with the permutation distributions of context id.
    Keeps one shuffled dprime, cluster and pvalue example.
    :param site: string  identifying the site
    :param probes: list of probe numbers
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

    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, condition_axis=2)
    clusters = get_clusters_mass(dprime, cluster_threshold, axis=-1)
    # shape Cell x CtxPair x Probe x Time

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle, then calculates the pairwise dprime

    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)

    # we need a single top quantil, and its a single value over all time points

    # we need a single top quantile, and it s a single value over all time points
    # context_pair*probe and neuron*context_pair*probe
    bf_corrections = dict(none=1,
                          bf_cp=len(ctx_pairs) * prb,
                          bf_ncp=chn * len(ctx_pairs) * prb)
    qnt_shape = list(dprime.shape)
    qnt_shape[-1] = 1
    quantiles = {corr_name: np.zeros(qnt_shape) for corr_name in bf_corrections.keys()}
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

        cpn_shuff_dprime = cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=3)
        del (shuf_trialR)

        # for the shuffles, get only the value of the biggest cluster
        cpn_shuf_clstr = get_clusters_mass(cpn_shuff_dprime, cluster_threshold, axis=-1)
        cpn_shuf_clstr_max = np.max(np.abs(cpn_shuf_clstr), axis=-1, keepdims=True)

        # finally calculates pvalues for each individual cluster
        real_clstr = clusters[:, ctx_pairs.index((c0, c1)), ...][:, None, ...]
        cpn_pval = _raw_pvalue(real_clstr, cpn_shuf_clstr_max)
        pvalue[:, cpn, :, :] = cpn_pval.squeeze(axis=1)

        # saves raw quantiles for display alongside multiple comparisons corrections for
        for corr_name, ncorr in bf_corrections.items():  # [0.05, 0.05 / 40, 0.05 / 550]:
            alpha = 0.05 / ncorr  # asumes initial alpha of 0.05
            quantiles[corr_name][:, cpn, :, :] = np.quantile(cpn_shuf_clstr_max, 1 - alpha, axis=0).squeeze(axis=1)

        # saves onle last single random shuffle example and its corresponding pvalue
        sf_eg = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)
        sf_eg_dprime = cDP.pairwise_dprimes(sf_eg, observation_axis=0, condition_axis=2)
        sf_eg_clust = get_clusters_mass(sf_eg_dprime, cluster_threshold, axis=-1)
        shuf_eg_dprime[:, cpn, :, :] = sf_eg_dprime.squeeze(axis=1)
        shuf_eg_clust[:, cpn, :, :] = sf_eg_clust.squeeze(axis=1)
        shuf_eg_pvalue[:, cpn, :, :] = _raw_pvalue(sf_eg_clust, cpn_shuf_clstr_max).squeeze(axis=1)
        del (cpn_shuff_dprime)

        if False and (ctx_pairs.index((0, 1)) == cpn):
            alpha = 0.05
            ncomp = prb * len(ctx_pairs)
            alpha_corr = alpha / ncomp

            from src.visualization.utils import squarefy
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
    clust_quant_pval = {'clusters': clusters, **quantiles, 'pvalue': pvalue}
    shuffled_eg = dict(dprime=shuf_eg_dprime, clusters=shuf_eg_clust, pvalue=shuf_eg_pvalue)

    return dprime, clust_quant_pval, goodcells, shuffled_eg


@memory.cache
def _load_probewise_dPCA_raster(site, contexts, probes, meta, load_fn=load_site_formated_raster):
    trialR, goodcells = load_fn(site, contexts, probes, **meta)

    rep, unt, ctx, prb, tme = trialR.shape
    if contexts == 'all':
        contexts = list(range(0, ctx))
    if probes == 'all':
        probes = list(range(1, prb + 1))

    R = cdPCA.get_centered_means(trialR)

    var_capt = list()
    trialZ = defaultdict(list)
    Z = defaultdict(list)
    pdpca = list()
    for pp in probes:
        probe_idx = probes.index(pp)
        probe_trialR = trialR[..., probe_idx, :]
        probe_R = R[..., probe_idx, :]

        # calculates dPCA considering all context categories
        probe_Z, probe_trialZ, dpca = cdPCA._cpp_dPCA(probe_R, probe_trialR)
        pdpca.append(dpca)
        var_capt.append(cdPCA.variance_captured(dpca, probe_R))

        for marg in dpca.marginalizations:
            trialZ[marg].append(probe_trialZ[marg])
            Z[marg].append(probe_Z[marg])

    for marg in dpca.marginalizations:
        trialZ[marg] = np.stack(trialZ[marg], axis=-2)
        Z[marg] = np.stack(Z[marg], axis=-2)

    return trialZ, Z, goodcells, pdpca, var_capt


@memory.cache
def probewise_dPCA_dprimes(site, contexts, probes, meta, load_fn=load_site_formated_raster):
    """
    performs dimensionality reduction with dPCA done independently for each probe. Then uses the first context dependent
    demixed principal component to calculated the dprime between context for all probes in the site. Calculates
    significance using montecarlo shuffling of the context identity.
    :param site: string  identifying the site
    :param probes: list of probe numbers
    :param meta: dict with meta parameters
    :return: dprime (ndarray with shape PC x Ctx_pair x Probe x Time),
             shuffled_dprimes (ndarray with shape Montecarlo x PC x Ctx_pair x Probe x Time),
             goocells (list of strings)
    """
    raise NotImplementedError('BROKEN STATISTICS!: ensure that you are calculating shuffles on a context-pair basis')
    trialR, goodcells = load_fn(site, contexts, probes, **meta)

    trialZ, _, goodcells, _, var_capt = _load_probewise_dPCA_raster(site, contexts, probes, meta, load_fn)
    dprime = cDP.pairwise_dprimes(trialZ['ct'][:, 0, ...], observation_axis=0, condition_axis=1)

    rep, unt, ctx, prb, tme = trialR.shape
    if contexts == 'all':
        contexts = list(range(0, ctx))
    if probes == 'all':
        probes = list(range(1, prb + 1))
    transition_pairs = list(itt.combinations(contexts, 2))

    # does the shuffling over individual probes since the original dPCA was done one probe at a time

    shuffled_dprime = np.empty([meta['montecarlo'], len(transition_pairs), prb, tme])
    for pp in probes:
        probe_idx = probes.index(pp)
        probe_trialR = trialR[..., probe_idx, :]

        # Shuffles the rasters n times and organizes in an array with the same shape as the original raster plus one
        # dimension with size meta['montecarlo'] containing each shuffle
        # calculates the pairwise dprime
        print(f"\nshuffling {meta['montecarlo']} times")
        rng = np.random.default_rng(42)

        shuf_projections = np.empty([meta['montecarlo'], rep, ctx, tme])
        shuf_projections[:] = np.nan

        shuf_trialR = probe_trialR.copy()
        for rr in range(meta['montecarlo']):
            shuf_trialR = shuffle(shuf_trialR, shuffle_axis=2, indie_axis=0, rng=rng)
            shuf_R = np.mean(shuf_trialR, axis=0)

            # saves the first regularizer to speed things up.
            if rr == 0:
                _, shuf_trialZ, shuf_dpca = cdPCA._cpp_dPCA(shuf_R, shuf_trialR)
                regularizer = shuf_dpca.regularizer
            else:
                _, shuf_trialZ, dpca = cdPCA._cpp_dPCA(shuf_R, shuf_trialR, {'regularizer': regularizer})

            shuf_projections[rr, :] = shuf_trialZ['ct'][:, 0, ...]

        shuffled_dprime[:, :, probe_idx, :] = cDP.pairwise_dprimes(shuf_projections,
                                                                   observation_axis=1,
                                                                   condition_axis=2)

    # add dimension for single PC, for compatibility with single cell arrays
    dprime = np.expand_dims(dprime, axis=0)
    shuffled_dprime = np.expand_dims(shuffled_dprime, axis=1)
    shuff_dprime_quantiles = _signif_quantiles(shuffled_dprime)

    return dprime, shuff_dprime_quantiles, goodcells, var_capt


@memory.cache
def _load_probewise_LDA_raster(site, contexts, probes, meta, load_fn=load_site_formated_raster):
    trialR, goodcells = load_fn(site, contexts, probes, **meta)

    rep, unt, ctx, prb, tme = trialR.shape
    if contexts == 'all':
        contexts = list(range(0, ctx))
    if probes == 'all':
        probes = list(range(1, prb + 1))

    # runs dimentionality reduction over each probe independently
    trialZ = list()
    transformations = list()
    var_capt = list()
    for pp in probes:
        probe_idx = probes.index(pp)
        probe_trialR = trialR[..., probe_idx, :]

        # calculates LDA considering all 4 categories
        LDA_projection, trans = cLDA.fit_transform_over_time(probe_trialR)
        LDA_projection = LDA_projection.squeeze(axis=1)  # shape Trial x Context x Time

        trialZ.append(LDA_projection)
        transformations.append(trans)

    trialZ = np.stack(trialZ, axis=2)  # shape Trial x Contexts x Probes x Time
    transformations = np.stack(transformations, axis=2)  # shape Neurons x lowDim x Probes x Time
    Z = trialZ.mean(axis=0)

    return trialZ, Z, goodcells, transformations, var_capt


@memory.cache
def probewise_LDA_dprimes(site, contexts, probes, meta, load_fn=load_site_formated_raster):
    """
    performs dimensionality reduction with LDA done independently for each probe. the uses the discriminated projection
    to calculate the dprime between context for all probes in the site. Calculates
    significance using montecarlo shuffling of the context identity.
    :param site: string  identifying the site
    :param probes: list of probe numbers
    :param meta: dict with meta parameters
    :return: dprime (ndarray with shape Ctx_pair x Probe x Time),
             shuffled_dprimes (ndarray with shape Montecarlo x Ctx_pair x Probe x Time),
             goocells (list of strings)
    """
    raise NotImplementedError('BROKEN STATISTICS!: ensure that you are calculating shuffles on a context-pair basis')
    trialR, goodcells = load_fn(site, contexts, probes, **meta)

    trialZ, _, _, transformations, _ = _load_probewise_LDA_raster(site, contexts, probes, meta, load_fn)

    rep, unt, ctx, prb, tme = trialR.shape
    if contexts == 'all':
        contexts = list(range(0, ctx))
    if probes == 'all':
        probes = list(range(1, prb + 1))

    transition_pairs = list(itt.combinations(contexts, 2))

    # iterates over each probe
    dprime = cDP.pairwise_dprimes(trialZ, observation_axis=0, condition_axis=1)

    shuffled_dprime = np.empty([meta['montecarlo'], len(transition_pairs), prb, tme])
    for pp in probes:
        probe_idx = probes.index(pp)
        probe_trialR = trialR[..., probe_idx, :]

        # Shuffles the rasters n times and organizes in an array with the same shape as the original raster plus one
        # dimension with size meta['montecarlo'] containing each shuffle
        # calculates the pairwise dprime
        print(f"\nshuffling {meta['montecarlo']} times")
        rng = np.random.default_rng(42)

        shuf_projections = np.empty([meta['montecarlo'], rep, ctx, tme])
        shuf_projections[:] = np.nan

        shuf_trialR = probe_trialR.copy()

        for rr in range(meta['montecarlo']):
            shuf_trialR = shuffle(shuf_trialR, shuffle_axis=2, indie_axis=0, rng=rng)
            shuf_LDA_projection, _ = cLDA.fit_transform_over_time(shuf_trialR)
            shuf_projections[rr, ...] = shuf_LDA_projection.squeeze(axis=1)  # shape Trial x Context x Time

        shuffled_dprime[:, :, probe_idx, :] = cDP.pairwise_dprimes(shuf_projections,
                                                                   observation_axis=1,
                                                                   condition_axis=2)
    # add dimension for single PC, for compatibility with single cell arrays
    dprime = np.expand_dims(dprime, axis=0)
    shuffled_dprime = np.expand_dims(shuffled_dprime, axis=1)
    shuff_dprime_quantiles = _signif_quantiles(shuffled_dprime)

    return dprime, shuff_dprime_quantiles, goodcells, None


@memory.cache
def _load_full_dPCA_raster(site, contexts, probes, meta, load_fn=load_site_formated_raster):
    trialR, goodcells = load_fn(site, contexts, probes, **meta)
    R = cdPCA.get_centered_means(trialR)

    # calculates full dPCA. i.e. considering all 4 categories
    Z, trialZ, dpca = cdPCA._cpp_dPCA(R, trialR)

    # calculates the variance explained. special case for full dpca, not present in other dprime approaches
    var_capt = cdPCA.variance_captured(dpca, R)

    return trialZ, Z, goodcells, dpca, var_capt


@memory.cache
def full_dPCA_dprimes(site, contexts, probes, meta, load_fn=load_site_formated_raster):
    raise NotImplementedError('BROKEN STATISTICS!: ensure that you are calculating shuffles on a context-pair basis')
    trialR, goodcells = load_fn(site, contexts, probes, **meta)
    R = cdPCA.get_centered_means(trialR)

    trialZ, Z, goodcells, dpca, var_capt = _load_full_dPCA_raster(site, contexts, probes, meta, load_fn)

    # calculates full dPCA. i.e. considering all 4 categories
    dPCA_projection = trialZ['ct'][:, 0, ...]
    dprime = cDP.pairwise_dprimes(dPCA_projection, observation_axis=0, condition_axis=1)

    # calculates the variance explained. special case for full dpca, not present in other dprime approaches
    var_capt = cdPCA.variance_captured(dpca, R)

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle
    # calculates the pairwise dprime
    rep, ctx, prb, tme = dPCA_projection.shape

    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)

    shuf_projections = np.empty([meta['montecarlo'], rep, ctx, prb, tme])

    shuf_trialR = trialR.copy()

    for rr in range(meta['montecarlo']):
        shuf_trialR = shuffle(shuf_trialR, shuffle_axis=2, indie_axis=0, rng=rng)
        shuf_R = np.mean(shuf_trialR, axis=0)

        # saves the first regularizer to speed things up.
        if rr == 0:
            _, shuf_trialZ, shuf_dpca = cdPCA._cpp_dPCA(shuf_R, shuf_trialR)
            regularizer = shuf_dpca.regularizer
        else:
            _, shuf_trialZ, dpca = cdPCA._cpp_dPCA(shuf_R, shuf_trialR, {'regularizer': regularizer})

        shuf_projections[rr, ...] = shuf_trialZ['ct'][:, 0, ...]

    shuffled_dprime = cDP.pairwise_dprimes(shuf_projections, observation_axis=1, condition_axis=2)

    # add dimension for single PC, for compatibility with single cell arrays
    dprime = np.expand_dims(dprime, axis=0)
    shuffled_dprime = np.expand_dims(shuffled_dprime, axis=1)
    shuff_dprime_quantiles = _signif_quantiles(shuffled_dprime)

    return dprime, shuff_dprime_quantiles, goodcells, var_capt


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.visualization.utils import squarefy

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

    cluster_threshold = 2
    (dprime, clust_quant_pval, goodcells, shuffled_eg), _ = single_cell_dprimes_cluster_mass.call(site=cellid,
                                                                                                  contexts='all',
                                                                                                  probes='all',
                                                                                                  cluster_threshold=cluster_threshold,
                                                                                                  meta=meta)

    pvalue = clust_quant_pval['pvalue']
    clusters = clust_quant_pval['clusters']

    alpha = 0.05
    ncomp = dprime.shape[2] * dprime.shape[1]
    alpha_corr = alpha / ncomp

    if dprime.shape[1] == 10:
        ctx = 5
    elif dprime.shape[1] == 55:
        ctx = 11
    else:
        raise ValueError('unknown nuber of context pairs')

    ctx_pairs = list(itt.combinations(range(ctx), 2))

    eg_idx = np.s_[goodcells.index(cellid), ctx_pairs.index((0, 1)), 3 - 1, :]
    # eg_idx = np.s_[goodcells.index(cellid), 9, 3-1, :]

    if np.sum(dprime[eg_idx]) < 0:
        flip = -1
    else:
        flip = 1

    eg_idx = np.s_[goodcells.index(cellid), ctx_pairs.index((0, 1)), 3 - 1, :]
    # eg_idx = np.s_[goodcells.index(cellid), 9, 3-1, :]

    if np.sum(dprime[eg_idx]) < 0:
        flip = -1
    else:
        flip = 1

    t = np.arange(30)
    d = dprime[eg_idx] * flip
    c = clusters[eg_idx] * flip
    # s = cpn_shuf_clstr_max[:, eg_idx[0], 0, eg_idx[2], 0]
    p = pvalue[eg_idx]

    tt, dd = squarefy(t, d)
    _, cc = squarefy(t, c)
    tt, pp = squarefy(t, p)

    ci_c = clust_quant_pval[f'{alpha_corr:.5f}'][eg_idx]
    ci = clust_quant_pval[f'{alpha:.5f}'][eg_idx]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=[8, 8])

    # raw data
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

    fig.show()

    # summed_time = np.sum(np.absolute(clusters), axis=-1)
    # sorted_egs = np.argsort(summed_time.flatten())
    # sorted_egs = np.unravel_index(sorted_egs, summed_time.shape)
    # sorted_egs = np.asarray(sorted_egs).T[::-1,:]
    # for eg_idx in sorted_egs:
    #     # eg_idx = np.unravel_index(np.argmax(np.absolute(clusters)), shape=clusters.shape)
    #     eg_idx = np.s_[eg_idx[0], eg_idx[1], eg_idx[2], :]
    #
    #     fig, ax = plt.subplots(figsize=[8, 8])
    #     ax.plot(dprime[eg_idx], label='dprime')
    #     ax2 = ax.twinx()
    #     max_clust = np.max(np.absolute(clusters[eg_idx]))
    #     # norm_clust = real_clstr[eg_idx] / max_clust
    #     ax.plot(clusters[eg_idx], label='cluster')
    #     ax2.plot(pvalue[eg_idx], label='pvalue', color='green')
    #     print(pvalue[eg_idx])
    #     # norm_shuf = cpn_shuf_clstr_max[(np.s_[:], )+eg_idx].squeeze() / max_clust
    #     ax.fill_between(np.arange(30), ci[eg_idx] * -1, ci[eg_idx], alpha=0.5, color='gray', label='shuf_max_clust')
    #     # ax.axhline(ci[eg_idx] * -1, alpha=0.1, color='gray')
    #     ax.axhline(cluster_threshold, color='red', linestyle=':', label='clust_threshold')
    #     ax.axhline(cluster_threshold * -1, color='red', linestyle=':')
    #     ax2.axhline(alpha, color='brown', linestyle='--', label='alpha')
    #     ax2.set_yscale('log')
    #     fig.legend()
    #     fig.show()
    #     print(' ')
