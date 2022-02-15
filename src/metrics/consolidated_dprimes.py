from collections import defaultdict
import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import numpy as np
from joblib import Memory

from src.data.rasters import load_site_formated_raster
from src.data import LDA as cLDA, dPCA as cdPCA
from src.metrics import dprime as cDP
from src.metrics.significance import _signif_quantiles, _raw_pvalue
from src.utils.tools import shuffle_along_axis as shuffle
from src.root_path import config_path

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

    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, condition_axis=2,
                                  flip=meta['dprime_absolute'])  # shape Cell x CtxPair x Probe x Time

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle, then calculates the pairwise dprime

    print(f"\nshuffling {meta['montecarlo']} times")
    rng = np.random.default_rng(42)

    # shuffling on a context pair basis for consistancy with the pairwise dprime, and as a bonus, to deal with memory issues
    ctx_pairs = list(itt.combinations(range(ctx),2))

    qnt_shape = (2, ) + dprime.shape
    quantiles = defaultdict(lambda : np.zeros(qnt_shape))
    pvalue = np.zeros_like(dprime)

    shuf_eg_dprime = np.zeros_like(dprime)
    shuf_eg_pvalue = np.zeros_like(dprime)

    for cpn, (c0, c1) in enumerate(ctx_pairs):
        print(f"    context pair {c0:02d}_{c1:02d}")
        shuf_trialR = np.empty((meta['montecarlo'], rep, chn, 2, prb, tme))
        ctx_shuffle = trialR[:, :, (c0,c1), :, :].copy() # tria, context, probe, time

        for rr in range(meta['montecarlo']):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)

        neur_shuff_dprime = cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=3,
                                               flip=meta['dprime_absolute'])

        # saves neuron pval for refined multiple comparisons with flexible alphas
        real_dprime = dprime[:,ctx_pairs.index((c0, c1)),...][:,None,...]
        pvalue[:,cpn,:,:] = _raw_pvalue(real_dprime, neur_shuff_dprime).squeeze(axis=1)

        # saves quantiles mostly for display, i.e. gray confidense intervale on dprime plots
        neur_shuff_quantils = _signif_quantiles(neur_shuff_dprime)
        for alpha, qntls in neur_shuff_quantils.items():
            quantiles[alpha][:,:,cpn,:,:] = qntls.squeeze(axis=2)


        # saves onle last single random shuffle example and its corresponding pvlaue
        sf_eg = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0, rng=rng)
        sf_eg_dprime = cDP.pairwise_dprimes(sf_eg, observation_axis=0, condition_axis=2,
                                                       flip=meta['dprime_absolute'])
        shuf_eg_dprime[:,cpn,:,:] = sf_eg_dprime.squeeze(axis=1)
        shuf_eg_pvalue[:,cpn,:,:] = _raw_pvalue(sf_eg_dprime, neur_shuff_dprime).squeeze(axis=1)

    # neat little output packages.
    pval_quantiles = {'pvalue':pvalue, **quantiles}
    shuffled_eg=dict(dprime=shuf_eg_dprime, pvalue=shuf_eg_pvalue)

    return dprime, pval_quantiles, goodcells, shuffled_eg


@memory.cache
def _load_probewise_dPCA_raster(site, contexts, probes, meta, load_fn=load_site_formated_raster):
    trialR, goodcells = load_fn(site, contexts, probes, **meta)

    rep, unt, ctx, prb, tme = trialR.shape
    if contexts == 'all':
        contexts = list(range(0, ctx))
    if probes == 'all':
        probes = list(range(1, prb+1))

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
    dprime = cDP.pairwise_dprimes(trialZ['ct'][:,0,...], observation_axis=0, condition_axis=1,
                                                       flip=meta['dprime_absolute'])


    rep, unt, ctx, prb, tme = trialR.shape
    if contexts == 'all':
        contexts = list(range(0, ctx))
    if probes == 'all':
        probes = list(range(1, prb+1))
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

            #saves the first regularizer to speed things up.
            if rr == 0:
                _, shuf_trialZ, shuf_dpca = cdPCA._cpp_dPCA(shuf_R, shuf_trialR)
                regularizer = shuf_dpca.regularizer
            else:
                _, shuf_trialZ, dpca = cdPCA._cpp_dPCA(shuf_R, shuf_trialR, {'regularizer':regularizer})

            shuf_projections[rr, :] = shuf_trialZ['ct'][:, 0, ...]

        shuffled_dprime[:, :, probe_idx, :] = cDP.pairwise_dprimes(shuf_projections,
                                                                   observation_axis=1,
                                                                   condition_axis=2,
                                                                   flip=meta['dprime_absolute'])

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

    trialZ = np.stack(trialZ, axis=2) # shape Trial x Contexts x Probes x Time
    transformations = np.stack(transformations, axis=2) # shape Neurons x lowDim x Probes x Time
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
    dprime = cDP.pairwise_dprimes(trialZ, observation_axis=0, condition_axis=1,
                                                       flip=meta['dprime_absolute'])


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
                                                                   condition_axis=2,
                                                                   flip=meta['dprime_absolute'])
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
    dprime = cDP.pairwise_dprimes(dPCA_projection, observation_axis=0, condition_axis=1,
                                  flip=meta['dprime_absolute'])

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

    shuffled_dprime = cDP.pairwise_dprimes(shuf_projections, observation_axis=1, condition_axis=2,
                                           flip=meta['dprime_absolute'])

    # add dimension for single PC, for compatibility with single cell arrays
    dprime = np.expand_dims(dprime, axis=0)
    shuffled_dprime = np.expand_dims(shuffled_dprime, axis=1)
    shuff_dprime_quantiles = _signif_quantiles(shuffled_dprime)

    return dprime, shuff_dprime_quantiles, goodcells, var_capt


if __name__ == "__main__":

    id = 'TNC010a' # A1 10 sounds
    # id = 'ARM022a' # PEG 4 sounds

    out = single_cell_dprimes(site=id,contexts='all',probes='all',
                        meta={'reliability': 0.1,  # r value
                              'smoothing_window': 0,  # ms
                              'raster_fs': 30,
                              'montecarlo': 1000,
                              'zscore': True,
                              'dprime_absolute': None,
                              'stim_type': 'permutations',
                              'alpha': 0.05}, )

    for ii in out:
        print(type(ii))

    pass





