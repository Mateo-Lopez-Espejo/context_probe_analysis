import matplotlib.pyplot as plt
import numpy as np
import itertools as itt
import pathlib as pl

from cpn_load import load
from reliability import signal_reliability
import cpn_dPCA as cdPCA

from cpp_cache import make_cache, get_cache

import cpn_LDA as cLDA
import cpn_dprime as cDP
from progressbar import ProgressBar

from tools import shuffle_along_axis as shuffle
import scipy.stats as sst

"""
This should nevere be used, the LDA will overfit every time bin thus generating a false hihg d' taht cannot be
reachede by the shuffled data transformed with the original LDA 
"""


def mean_confidence_interval(array, confidence=0.95, axis=0):
    '''
    calculates the mean and confidence interval of an array
    :param array:
    :param confidence:
    :return:
    '''
    n = array.shape[axis]
    m, se, std = np.mean(array, axis=axis), sst.sem(array, axis=axis), np.std(array, axis=axis)
    h = se * sst.t.ppf((1 + confidence) / 2., n - 1) # ToDo check if this formula is adecuate
    return m, h


def cint(array, confidence, x=None, ax=None, fillkwargs={}):
    if ax is None:
        fig, ax = plt.subplots()

    if x is None:
        x = np.arange(0, array.shape[0], 1)


    # lower, upper = mean_confidence_interval(array, confidence, axis=1)

    tails = (1-confidence)/2
    low = tails * 100
    high = (1-tails) * 100
    lower, upper = np.percentile(array,[low,high], axis=1)

    ax.fill_between(x, lower, upper, **fillkwargs)

    return ax


def LDA_halfMC_fourway_analysis(site, probe, meta):
    recs = load(site)

    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
    sig = rec['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = cdPCA.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                   smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'])

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, _, _ = cdPCA.format_raster(raster)
    trialR = trialR.squeeze()  # squeezes out probe
    R, C, S, T = trialR.shape

    # calculates full LDA. i.e. considering all 4 categories
    LDA_projection, LDA_transformation = cLDA.fit_transform_over_time(trialR, 1)
    dprime= cDP.pairwise_dprimes(LDA_projection.squeeze())

    # calculates floor (ctx shuffle) and ceiling (simulated data)
    sim_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))
    shuf_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))

    ctx_shuffle = trialR.copy()

    pbar = ProgressBar()
    for rr in pbar(range(meta['montecarlo'])):

        # ceiling: simulates data, calculates dprimes
        sim_trial = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0),
                                      size=[R, C, S, T])
        sim_projection = cLDA.transform_over_time(cLDA._reorder_dims(sim_trial), LDA_transformation)
        sim_dprime[rr, ...] = cDP.pairwise_dprimes(cLDA._recover_dims(sim_projection).squeeze())

        ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)
        shuf_projection = cLDA.transform_over_time(cLDA._reorder_dims(ctx_shuffle), LDA_transformation)
        shuf_dprime[rr, ...] = cDP.pairwise_dprimes(cLDA._recover_dims(shuf_projection).squeeze())

    return dprime, shuf_dprime, sim_dprime


def LDA_halfMC_twoway_analysis(site, probe, meta):
    recs = load(site)

    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
    sig = rec['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # outer lists to save the dprimes foe each pair of ctxs
    dprime = list()
    shuf_dprime = list()
    sim_dprime = list()

    for transitions in itt.combinations(meta['transitions'], 2):

        # get the full data raster Context x Probe x Rep x Neuron x Time
        raster = cdPCA.raster_from_sig(sig, probe, channels=goodcells, transitions=transitions,
                                       smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'])

        # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
        trialR, _, _ = cdPCA.format_raster(raster)
        trialR = trialR.squeeze()  # squeezes out probe
        R, C, S, T = trialR.shape

        # calculates full LDA. i.e. considering all 4 categories
        LDA_projection, LDA_transformation = cLDA.fit_transform_over_time(trialR, 1)
        dp = cDP.pairwise_dprimes(LDA_projection.squeeze())
        dprime.append(dp)

        # calculates floor (ctx shuffle) and ceiling (simulated data)
        sim_dp = np.empty([meta['montecarlo']] + list(dp.shape))
        shuf_dp = np.empty([meta['montecarlo']] + list(dp.shape))

        ctx_shuffle = trialR.copy()

        pbar = ProgressBar()
        for rr in pbar(range(meta['montecarlo'])):
            # ceiling: simulates data, calculates dprimes
            sim_trial = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0),
                                         size=[R, C, S, T])
            sim_projection = cLDA.transform_over_time(cLDA._reorder_dims(sim_trial), LDA_transformation)
            sim_dp[rr, ...] = cDP.pairwise_dprimes(cLDA._recover_dims(sim_projection).squeeze())

            ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)
            shuf_projection = cLDA.transform_over_time(cLDA._reorder_dims(ctx_shuffle), LDA_transformation)
            shuf_dp[rr, ...] = cDP.pairwise_dprimes(cLDA._recover_dims(shuf_projection).squeeze())


        shuf_dprime.append(shuf_dp)
        sim_dprime.append(sim_dp)

    # orders the list into arrays of the same shape as the fourwise analysis: MonteCarlo x Pair x Time

    dprime = np.concatenate(dprime, axis=0)
    shuf_dprime = np.concatenate(shuf_dprime, axis=1)
    sim_dprime = np.concatenate(sim_dprime, axis=1)

    return dprime, shuf_dprime, sim_dprime


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628', # blue, orange, green, brow,
                  '#984ea3', '#999999', '#e41a1c', '#dede00'] # purple, gray, scarlet, lime

trans_color_map = {'silence': '#377eb8', # blue
                   'continuous': '#ff7f00', # orange
                   'similar': '#4daf4a', # green
                   'sharp': '#a65628'} # brown

ci_color = {'shuffled': '#dede00',
            'simulated': '#984ea3'}

meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 0, # ms
        'raster_fs': 30,
        'transitions' : ['silence', 'continuous', 'similar', 'sharp'],
        'significance': False,
        'montecarlo': 1000 }

analysis_name = 'LDA_dprime'
analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
code_to_name = {'t': 'Probe', 'ct': 'Context'}


for site, probe in zip(['AMT029a', 'ley070a'],[5,2]):

    fourway_name = f'191014_{site}_P{probe}_fourway_analysis'

    fourway = make_cache(function= LDA_halfMC_fourway_analysis,
                         func_args= {'site': site, 'probe': probe, 'meta': meta},
                         classobj_name=fourway_name,
                         cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}')

    twoway_name = f'191014_{site}_P{probe}_twoway_analysis'

    twoway = make_cache(function=LDA_halfMC_twoway_analysis,
                        func_args={'site': site, 'probe': probe, 'meta': meta},
                        classobj_name=twoway_name,
                        cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}')


    four, two=get_cache(fourway), get_cache(twoway)

    for analysis, a_name in zip([four, two], ['fourway', 'paired']):

        real, shuffled, simulated = analysis

        # pairwise dprime
        trans_pairs = list(itt.combinations(meta['transitions'], 2))
        trans_count = 0

        nrow = 2
        ncol = 3
        fig, axes = plt.subplots(nrow,ncol, sharey=True, sharex=True, squeeze=False)
        for row, col in itt.product(range(nrow), range(ncol)):
            trans_pair = trans_pairs[trans_count]

            ax = axes[row, col]
            T = real.shape[1]
            time = np.linspace(0,T/meta['raster_fs'], T, endpoint=False)

            ci = 0.9
            ax.plot(time, np.mean(shuffled[:, trans_count, :], axis=0), color=ci_color['shuffled'], alpha=1)
            cint(shuffled[:, trans_count, :].T, ci, x=time, ax=ax,
                 fillkwargs={'alpha': 0.5, 'color': ci_color['shuffled'], 'label':'id shuffle'})
            # ceiling
            ax.plot(time, np.mean(simulated[:, trans_count, :], axis=0), color=ci_color['simulated'], alpha=1)
            cint(simulated[:, trans_count, :].T, ci, x=time, ax=ax,
                 fillkwargs={'alpha': 0.5, 'color': ci_color['simulated'], 'label':'decorrelated'})
            # real dprime
            ax.plot(time, real[trans_count, :], color='black', linestyle='-', label='real value')

            # Formatting
            ax.tick_params(labelsize='15')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            ax.tick_params(labelsize='15')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # distance labels only on leftmost plots
            if col == 0:
                ax.set_ylabel(f"d'", fontsize=20)
                # trial_ax.set_ylabel(f'euclidean distance(Hz)', fontsize=20)
                pass
            else:
                ax.axes.get_yaxis().set_visible(False)

            ax.set_title(f'{trans_pair[0]} vs {trans_pair[1]}', fontsize=20)

            # time only in bottom plot
            if row != nrow-1:
                ax.axes.get_xaxis().set_visible(False)
            else:
                ax.set_xlabel('time (s)', fontsize=20)

            # adds legend in the last ax
            if trans_count+1 == nrow*ncol:
                ax.legend()

            trans_count += 1

        suptitle = f"{site} probe {probe} {a_name} LDA_halfMC dprime {meta['raster_fs']}Hz"
        fig.suptitle(suptitle, fontsize=20)

        analysis = f"LDA_halfMC_{a_name}_{meta['raster_fs']}Hz"

        # set figure to full size in tenrec screen
        fig.set_size_inches(19.2, 9.79)

        root = pl.Path(f'/home/mateo/Pictures/APAM/{analysis}')
        if not root.exists(): root.mkdir(parents=True, exist_ok=True)
        png = root.joinpath(suptitle).with_suffix('.png')
        fig.savefig(png, transparent=True, dpi=100)
        svg = png = root.joinpath(suptitle).with_suffix('.svg')
        fig.savefig(svg, transparent=True)