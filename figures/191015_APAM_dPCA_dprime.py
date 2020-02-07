import cpn_triplets as tp
from cpn_load import load
from reliability import signal_reliability

import cpn_dPCA as cdPCA

import matplotlib.pyplot as plt
import numpy as np
import itertools as itt
import pathlib as pl

from cpn_load import load
from reliability import signal_reliability
import cpn_dPCA as cdPCA
import cpn_dispersion as ndisp
from cpp_cache import make_cache, get_cache
import fancy_plots as plot
import cpn_triplets as tp
import joblib as jl

import cpn_LDA as cLDA
import cpn_dprime as cDP
from progressbar import ProgressBar

from cpp_PCA import PSTH_PCA as pca
from cpn_shuffle import shuffle_along_axis as shuffle
import scipy.stats as sst


"""
given a site and probe, calculates the d' over the dPCA projection for the probe responses.
includes the 95% confidence intervals for a context discrimination shuffle test (green) and
test for population code on context discrimination (purple)

the inner loop of the script cachese the real d prime as well as the n-fold Monte Carlo calculations
for the confidence intervals.

the outer loop can be run over all combination of (site, probe) but is set up to plot only two
example sites/probes 
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
    h = se * sst.t.ppf((1 + confidence) / 2., n - 1)  # ToDo check if this formula is adecuate
    return m, h


def cint(array, confidence, x=None, ax=None, fillkwargs={}):
    if ax is None:
        fig, ax = plt.subplots()

    if x is None:
        x = np.arange(0, array.shape[0], 1)

    # lower, upper = mean_confidence_interval(array, confidence, axis=1)

    tails = (1 - confidence) / 2
    low = tails * 100
    high = (1 - tails) * 100
    lower, upper = np.percentile(array, [low, high], axis=1)

    ax.fill_between(x, lower, upper, **fillkwargs)

    return ax


def dPCA_fourway_analysis(site, probe, meta):
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
                                   smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                   zscore=meta['zscore'])

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    trialR, R = trialR.squeeze(), R.squeeze()  # squeezes out probe
    Re, C, S, T = trialR.shape

    # calculates full dPCA. i.e. considering all 4 categories
    def fit_transformt(R, trialR):
        _, dPCA_projection, _, dpca = cdPCA.trials_dpca(R, trialR, significance=False, dPCA_parms={})
        dPCA_projection = dPCA_projection['ct'][:, 0, ]
        dPCA_transformation = np.tile(dpca.D['ct'][:, 0][:, None, None], [1, 1, T])
        return dPCA_projection, dPCA_transformation

    dPCA_projection, dPCA_transformation = fit_transformt(R, trialR)
    dprime = cDP.pairwise_dprimes(dPCA_projection)

    # calculates floor (ctx shuffle) and ceiling (simulated data)
    sim_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))
    shuf_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))

    ctx_shuffle = trialR.copy()

    pbar = ProgressBar()
    for rr in pbar(range(meta['montecarlo'])):
        # ceiling: simulates data, calculates dprimes
        sim_trial = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0),
                                     size=[Re, C, S, T])
        sim_projection = cLDA.transform_over_time(cLDA._reorder_dims(sim_trial), dPCA_transformation)
        sim_dprime[rr, ...] = cDP.pairwise_dprimes(cLDA._recover_dims(sim_projection).squeeze())

        ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)
        shuf_projection = cLDA.transform_over_time(cLDA._reorder_dims(ctx_shuffle), dPCA_transformation)
        shuf_dprime[rr, ...] = cDP.pairwise_dprimes(cLDA._recover_dims(shuf_projection).squeeze())

    return dprime, shuf_dprime, sim_dprime


def dPCA_twoway_analysis(site, probe, meta):
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
                                       smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                       zscore=meta['zscore'])

        # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
        trialR, R, _ = cdPCA.format_raster(raster)
        trialR, R = trialR.squeeze(), R.squeeze()  # squeezes out probe
        Re, C, S, T = trialR.shape

        # calculates full dPCA. i.e. considering all 4 categories
        def fit_transformt(R, trialR):
            _, dPCA_projection, _, dpca = cdPCA.trials_dpca(R, trialR, significance=False, dPCA_parms={})
            dPCA_projection = dPCA_projection['ct'][:, 0, ]
            dPCA_transformation = np.tile(dpca.D['ct'][:, 0][:, None, None], [1, 1, T])
            return dPCA_projection, dPCA_transformation

        dPCA_projection, dPCA_transformation = fit_transformt(R, trialR)
        dp = cDP.pairwise_dprimes(dPCA_projection)
        dprime.append(dp)

        # calculates floor (ctx shuffle) and ceiling (simulated data)
        sim_dp = np.empty([meta['montecarlo']] + list(dp.shape))
        shuf_dp = np.empty([meta['montecarlo']] + list(dp.shape))

        ctx_shuffle = trialR.copy()

        pbar = ProgressBar()
        for rr in pbar(range(meta['montecarlo'])):
            # ceiling: simulates data, calculates dprimes
            sim_trial = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0),
                                         size=[Re, C, S, T])
            sim_projection = cLDA.transform_over_time(cLDA._reorder_dims(sim_trial), dPCA_transformation)
            sim_dp[rr, ...] = cDP.pairwise_dprimes(cLDA._recover_dims(sim_projection).squeeze())

            ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)
            shuf_projection = cLDA.transform_over_time(cLDA._reorder_dims(ctx_shuffle), dPCA_transformation)
            shuf_dp[rr, ...] = cDP.pairwise_dprimes(cLDA._recover_dims(shuf_projection).squeeze())

        shuf_dprime.append(shuf_dp)
        sim_dprime.append(sim_dp)

    # orders the list into arrays of the same shape as the fourwise analysis: MonteCarlo x Pair x Time

    dprime = np.concatenate(dprime, axis=0)
    shuf_dprime = np.concatenate(shuf_dprime, axis=1)
    sim_dprime = np.concatenate(sim_dprime, axis=1)

    return dprime, shuf_dprime, sim_dprime


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',  # blue, orange, green, brow,
                  '#984ea3', '#999999', '#e41a1c', '#dede00']  # purple, gray, scarlet, lime

trans_color_map = {'silence': '#377eb8',  # blue
                   'continuous': '#ff7f00',  # orange
                   'similar': '#4daf4a',  # green
                   'sharp': '#a65628'}  # brown

ci_color = {'shuffled': 'orange',
            'simulated': 'purple'}

# transferable plotting parameters
plt.rcParams['svg.fonttype'] = 'none'
sup_title_size = 15
sub_title_size = 12
ax_lab_size = 15
ax_val_size = 11

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'significance': False,
        'montecarlo': 1000,
        'zscore': False}

analysis_name = 'LDA_dprime'
analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
code_to_name = {'t': 'Probe', 'ct': 'Context'}

for site, probe in zip(['AMT029a', 'ley070a'], [5, 2]):

    fourway_name = f'191015_{site}_P{probe}_fourway_analysis'

    fourway = make_cache(function=dPCA_fourway_analysis,
                         func_args={'site': site, 'probe': probe, 'meta': meta},
                         classobj_name=fourway_name,
                         cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}')

    # twoway_name = f'191015_{site}_P{probe}_twoway_analysis'
    #
    # twoway = make_cache(function= dPCA_twoway_analysis,
    #                      func_args={'site': site, 'probe': probe, 'meta': meta},
    #                      classobj_name=twoway_name,
    #                      cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}')

    four = get_cache(fourway)  # , get_cache(twoway)

    for analysis, a_name in zip([four], ['fourway']):

        real, shuffled, simulated = analysis

        # pairwise dprime
        trans_pairs = list(itt.combinations(meta['transitions'], 2))
        trans_count = 0

        nrow = 2
        ncol = 3
        fig, axes = plt.subplots(nrow, ncol, sharey=True, sharex=True, squeeze=False)
        for row, col in itt.product(range(nrow), range(ncol)):
            trans_pair = trans_pairs[trans_count]

            ax = axes[row, col]
            T = real.shape[1]
            time = np.linspace(0, T / meta['raster_fs'], T, endpoint=False)

            ci = 0.9
            ax.plot(time, np.mean(shuffled[:, trans_count, :], axis=0), color=ci_color['shuffled'], alpha=1)
            cint(shuffled[:, trans_count, :].T, ci, x=time, ax=ax,
                 fillkwargs={'alpha': 0.5, 'color': ci_color['shuffled'], 'label': 'context discrimination'})
            # ceiling
            ax.plot(time, np.mean(simulated[:, trans_count, :], axis=0), color=ci_color['simulated'], alpha=1)
            cint(simulated[:, trans_count, :].T, ci, x=time, ax=ax,
                 fillkwargs={'alpha': 0.5, 'color': ci_color['simulated'], 'label': 'population effect'})
            # real dprime
            ax.plot(time, real[trans_count, :], color='black', linestyle='-', label='real value')

            # Formatting
            ax.tick_params(labelsize=ax_val_size)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            ax.tick_params(labelsize=ax_val_size)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # distance labels only on leftmost plots
            if col == 0:
                ax.set_ylabel(f"d'", fontsize=ax_lab_size)
                # trial_ax.set_ylabel(f'euclidean distance(Hz)', fontsize=20)
                pass
            else:
                ax.axes.get_yaxis().set_visible(False)

            ax.set_title(f'{trans_pair[0]} vs {trans_pair[1]}', fontsize=sub_title_size)

            # time only in bottom plot
            if row != nrow - 1:
                ax.axes.get_xaxis().set_visible(False)
            else:
                ax.set_xlabel('time (s)', fontsize=ax_lab_size)

            # adds legend in the last ax
            if trans_count + 1 == nrow * ncol:
                ax.legend()

            trans_count += 1

        suptitle = f"{site} probe {probe} {a_name} dPCA dprime {meta['raster_fs']}Hz zscore {meta['zscore']}"
        fig.suptitle(suptitle, fontsize=sup_title_size)

        analysis = f"dPCA_{a_name}_{meta['raster_fs']}Hz_zscore-{meta['zscore']}"

        # set figure to full size in tenrec screen
        fig.set_size_inches(7, 4)
        root = pl.Path(f'/home/mateo/Pictures/APAM/final/{analysis}')
        if not root.exists(): root.mkdir(parents=True, exist_ok=True)
        png = root.joinpath(suptitle).with_suffix('.png')
        fig.savefig(png, transparent=True, dpi=100)
        svg = png = root.joinpath(suptitle).with_suffix('.svg')
        fig.savefig(svg, transparent=True)
