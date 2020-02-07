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

from cpn_shuffle import shuffle_along_axis as shuffle
from scipy.stats import ranksums, wilcoxon
import pandas as pd
import seaborn as sn

import collections as col

"""
Summary of the d' context discrimination significance, and propulation effect significance across all combinations of 
sites and probes.
The two metrics extracted are the total number of significant time bins and the position of the last time bin.

it is highly recomended to add a way of keeping track of the distibution of significant bins over time across each
category
"""


def fourway_analysis(site, probe, meta):
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
    trialR, _, _ = cdPCA.format_raster(raster)
    trialR = trialR.squeeze()  # squeezes out probe
    R, C, S, T = trialR.shape

    # calculates full LDA. i.e. considering all 4 categories
    LDA_projection, LDA_transformation = cLDA.fit_transform_over_time(trialR, 1)
    dprime = cDP.pairwise_dprimes(LDA_projection.squeeze())

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
        shuf_projection, _ = cLDA.fit_transform_over_time(ctx_shuffle)
        shuf_dprime[rr, ...] = cDP.pairwise_dprimes(shuf_projection.squeeze())

    return dprime, shuf_dprime, sim_dprime


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


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',  # blue, orange, green, brow,
                  '#984ea3', '#999999', '#e41a1c', '#dede00']  # purple, gray, scarlet, lime

trans_color_map = {'silence': '#377eb8',  # blue
                   'continuous': '#ff7f00',  # orange
                   'similar': '#4daf4a',  # green
                   'sharp': '#a65628'}  # brown

MC_color = {'shuffled': 'orange',
            'simulated': 'purple'}

# transferable plotting parameters
plt.rcParams['svg.fonttype'] = 'none'
sup_title_size = 30
sub_title_size = 20
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

all_probes = [2, 3, 5, 6]

# all_probes  = [2,3]
# all_probes  = [5,6]


all_sites = ['ley070a',  # good site. A1
             'ley072b',  # Primary looking responses with strong contextual effects
             'AMT028b',  # good site
             'AMT029a',  # Strong response, somehow visible contextual effects
             'AMT030a',  # low responses, Ok but not as good
             # 'AMT031a', # low response, bad
             'AMT032a']  # great site. PEG

bad_sites = list()

df = list()

# for site, probe in zip(['AMT029a', 'ley070a'],[5,2]):
for site, probe in itt.product(all_sites, all_probes):

    try:
        LDA_anal_name = f'191014_{site}_P{probe}_fourway_analysis'

        LDA_anal = make_cache(function=fourway_analysis,
                              func_args={'site': site, 'probe': probe, 'meta': meta},
                              classobj_name=LDA_anal_name,
                              cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}')
        # LDA_real, LDA_shuffled, LDA_simulated = get_cache(LDA_anal)

    except:
        bad_sites.append(f"{site}_P{probe}_LDA")
        continue

    try:
        dPCA_anal_name = f'191015_{site}_P{probe}_fourway_analysis'

        dPCA_anal = make_cache(function=dPCA_fourway_analysis,
                               func_args={'site': site, 'probe': probe, 'meta': meta},
                               classobj_name=dPCA_anal_name,
                               cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}')
        # dPCA_real, dPCA_shuffled, dPCA_simulated = get_cache(dPCA_anal)

    except:
        bad_sites.append(f"{site}_P{probe}_dPCA")
        continue

    for transformation, cache in zip(['LDA', 'dPCA'], [LDA_anal, dPCA_anal]):

        real, shuffled, simulated = get_cache(cache)

        for montecarlo, MCarray in zip(['context discrimination', 'population effect'], [shuffled, simulated]):

            # calculates a signed pvalue, the signs is indicative of the direction, with possitive being higher and
            # negative being lower than the mean of the Montecarlo distribution. Bi virtue of this distinction
            # the p value is calculated on a single tail.

            mont_mean = np.mean(MCarray, axis=0)

            neg_pval = np.sum((MCarray < real), axis=0) / meta['montecarlo']
            pos_pval = np.sum((MCarray > real), axis=0) / meta['montecarlo']

            pvalues = np.where(real >= mont_mean, pos_pval, -neg_pval)

            for pp, trans_pair in enumerate(itt.combinations(meta['transitions'], 2)):

                for pval_threshold in [0.05, 0.01, 0.001]:
                    signif = np.abs(pvalues[pp, :]) < pval_threshold
                    total_sig = np.sum(signif) * 100 / len(signif)  # todo remove the hardcode percentage
                    try:
                        last_sig = np.max(np.argwhere(signif))
                    except:  # if there is not significant
                        last_sig = 0

                    d = {'site': site,
                         'probe': probe,
                         'transformation': transformation,
                         'montecarlo': montecarlo,
                         'pair': f'{trans_pair[0]}_{trans_pair[1]}',
                         'threshold': pval_threshold,
                         'parameter': 'total_sig',
                         'value': total_sig}

                    df.append(d)

                    d = {'site': site,
                         'probe': probe,
                         'transformation': transformation,
                         'montecarlo': montecarlo,
                         'pair': f'{trans_pair[0]}_{trans_pair[1]}',
                         'threshold': pval_threshold,
                         'parameter': 'last_sig',
                         'value': last_sig}
                    df.append(d)

            # # plots to check proper pval calculation
            # fig, axes  = plt.subplots(2,3, sharex=True, sharey=True)
            # axes = np.ravel(axes)
            # for pp in range(real.shape[0]):
            #     ax = axes[pp]
            #     ax.plot(real[pp,:], color='black')
            #     ax.plot(MCarray[:,pp,:].T, color='gray', alpha=0.01)
            #     ax.plot(np.abs(pvalues[pp,:]), color='green')
            #     ax.plot(np.abs(pvalues[pp,:])<0.05, color='orange')
            #

DF = pd.DataFrame(df)
DF['area'] = ['A1' if site[0:3] == 'ley' else 'PEG' for site in DF.site]
DF['unique'] = [f'{site}_P{probe}_{pair}' for site, probe, pair in zip(DF.site, DF.probe, DF.pair)]
DF['trans_area'] = [f'{area} {trans}' for trans, area in zip(DF.transformation, DF.area)]

ff_thres = DF.threshold == 0.01
ff_total = DF.parameter == 'total_sig'
ff_last = DF.parameter == 'last_sig'

filtered = DF.loc[ff_thres & ff_total, :]

pallette = sn.set_palette(['orange', 'purple'])  # shuffled yellow, simulated purple
fig, ax = plt.subplots()
ax = sn.swarmplot(x='trans_area', y='value', hue='montecarlo', data=filtered, palette=pallette, dodge=True)

ax.set_title('total significant bins at p<0.01')
ax.set_ylabel('Significant bins (%)')
ax.set_xlabel('brain area - transformation')

# set figure to full size in tenrec screen
fig.set_size_inches(7, 4.5)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=ax_val_size)
ax.title.set_size(sub_title_size)
ax.xaxis.label.set_size(ax_lab_size)
ax.yaxis.label.set_size(ax_lab_size)

root = pl.Path(f'/home/mateo/Pictures/APAM')
if not root.exists(): root.mkdir(parents=True, exist_ok=True)
png = root.joinpath(f'paired_distance_summary').with_suffix('.png')
fig.savefig(png, transparent=True, dpi=100)
svg = png = root.joinpath(f'paired_distance_summary').with_suffix('.svg')
fig.savefig(svg, transparent=True)

# calculates significance tests
area_comp = col.defaultdict(dict)
trans_comp = col.defaultdict(dict)
for montecarlo in DF.montecarlo.unique():

    ff_mont = DF.montecarlo == montecarlo

    # compares areas iterates overtrans
    for trans in DF.transformation.unique():
        ff_trans = DF.transformation == trans
        filtered = DF.loc[ff_thres & ff_total & ff_trans & ff_mont, :]
        A1 = filtered.loc[
                 filtered.area == 'A1', 'value'].values * 30 / 100  # todo eliminate the hardcode back into count
        PEG = filtered.loc[filtered.area == 'PEG', 'value'].values * 30 / 100

        _, area_comp[montecarlo][trans] = ranksums(A1, PEG)

    # compares trasnformations  iterates over areas
    for area in DF.area.unique():
        ff_area = DF.area == area
        filtered = DF.loc[ff_thres & ff_total & ff_area & ff_mont, :]
        pivoted = filtered.pivot(columns='transformation', index='unique', values='value').values * 30 / 100

        _, trans_comp[montecarlo][area] = wilcoxon(pivoted[:, 0], pivoted[:, 1])

print(area_comp)
print(trans_comp)
