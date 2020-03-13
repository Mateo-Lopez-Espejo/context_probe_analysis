import itertools as itt

import matplotlib.pyplot as plt
import numpy as np
from progressbar import ProgressBar

import cpn_LDA as cLDA
import cpn_dPCA as cdPCA
import cpn_dprime as cDP
from cpn_load import load
from cpp_cache import make_cache, get_cache
from reliability import signal_reliability
from tools import shuffle_along_axis as shuffle

import fancy_plots as fplt

# todo edit motive
"""
Summary of the d' context discrimination significance, and propulation effect significance across all combinations of 
sites and probes.
The two metrics extracted are the total number of significant time bins and the position of the last time bin.

it is highly recomended to add a way of keeping track of the distibution of significant bins over time across each
category
"""

def cell_dprime(site, probe, meta):
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
                                   zscore=meta['zscore'], part='probe')

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    trialR, R = trialR.squeeze(), R.squeeze()  # squeezes out probe

    rep, chn, ctx, tme = trialR.shape


    trans_pairs = [f'{x}_{y}' for x, y in itt.combinations(meta['transitions'], 2)]

    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, conditiont_axis=2)  # shape CellPair x Cell x Time

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle

    shuffled = list()
    pbar = ProgressBar()
    print(f"\nshuffling {meta['montecarlo']} times")
    for tp in pbar(trans_pairs):
        shuf_trialR = np.empty([meta['montecarlo'], rep, chn, 2, tme])
        shuf_trialR[:] = np.nan

        tran_idx = np.array([meta['transitions'].index(t) for t in tp.split('_')])
        ctx_shuffle = trialR[:,:,tran_idx, :].copy()

        for rr in range(meta['montecarlo']):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)

        shuffled.append(cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, conditiont_axis=3))

    shuffled = np.stack(shuffled, axis=1)# shape Montecarlo x ContextPair x Cell x Time

    return dprime, shuffled, goodcells, trans_pairs


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
        'montecarlo': 1000,
        'zscore': False}

analysis_name = 'LDA_dprime'
analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
code_to_name = {'t': 'Probe', 'ct': 'Context'}

all_probes = [2, 3, 5, 6]

sites = ['ley070a',  # good site. A1
         'ley072b',  # Primary looking responses with strong contextual effects
         'AMT028b',  # good site
         'AMT029a',  # Strong response, somehow visible contextual effects
         'AMT030a',  # low responses, Ok but not as good
         # 'AMT031a', # low response, bad
         'AMT032a']  # great site. PEG

# for site, probe in zip(['AMT029a', 'ley070a'],[5,2]):
# all_sites = ['AMT029a']
# all_sites = ['AMT032a']
# all_probes = [5]

bad_sites = list()
all_pvalues = dict()
all_reals = dict()
all_shuffled = dict()


for site in sites:

    this_site_reals = list()
    this_site_shuffled = list()
    this_site_pvalues = list()
    for pp, probe in enumerate(all_probes):

        object_name = f'200221_{site}_P{probe}_single_cell_dprime'
        analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
        analysis_name = 'NTI_singel_cell_dprime'

        try:
            cache = make_cache(function=cell_dprime,
                               func_args={'site': site, 'probe': probe, 'meta': meta},
                               classobj_name=object_name,
                               cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}',
                               recache=False)

            real, shuffled, cell_names, trans_pairs = get_cache(cache)

        except:
            bad_sites.append(f"{site}_P{probe}_dPCA")
            continue

        this_site_reals.append(real)
        this_site_shuffled.append(shuffled)

        # single tailed p value base on the montecarlo shuffling

        pvalues = np.sum((shuffled >= real), axis=0) / meta['montecarlo']

        this_site_pvalues.append(pvalues)

    this_site_reals = np.stack(this_site_reals, axis=0)
    this_site_shuffled = np.stack(this_site_shuffled, axis=0)
    this_site_pvalues = np.stack(this_site_pvalues, axis=0)

    # reorders date in dictionary of cells
    for cc, cell in enumerate(cell_names):
        all_reals[cell] = this_site_reals[:, :, cc, :]
        all_shuffled[cell] = this_site_shuffled[:, :, :, cc, :].swapaxes(0, 1)
        all_pvalues[cell] = this_site_pvalues[:, :, cc, :]

# stacks the site individual arrays along a new site dimension. since the sites have disimilar cell number, pads
all_cells = np.array(list(all_pvalues.keys()))
all_sites = np.array([cell[0:7] for cell in all_cells])
all_regions = np.array([cell[0:3] for cell in all_cells])

all_types = list()
for t0, t1 in itt.combinations(meta['transitions'], 2):
    pair = f'{t0}_{t1}'
    if pair == 'silence_continuous':
        all_types.append('hard')
    elif pair == 'silence_similar':
        all_types.append('hard')
    elif pair == 'silence_sharp':
        all_types.append('soft')
    elif pair == 'continuous_similar':
        all_types.append('soft')
    elif pair == 'continuous_sharp':
        all_types.append('hard')
    elif pair == 'similar_sharp':
        all_types.append('hard')
    else:
        continue

threshold = 0.05
all_signif = {key: (val <= threshold) for key, val in all_pvalues.items()}
sig_array = np.stack(list(all_signif.values()), axis=0) # dimensions: Cell x Probe x trans_pair x time


# dimensions to collapse per cell: Probe, Transition.
# collapse one, then the other, then both, per cell
# calculates the bins over/under a significant threshold


# for each cell in each site collapses across probes and context pairs
for site in sites:
    collapsed = np.sum(sig_array[all_sites == site, :, :, :], axis=(1,2))
    site_cells = all_cells[all_sites == site]

    fig, axes = fplt.subplots_sqr(collapsed.shape[0], sp_kwargs={'sharey':True})
    for ax, hist, cell in zip(axes, collapsed, site_cells ):
        ax.bar(np.arange(collapsed.shape[-1]), hist, align='edge')
        ax.set_title(cell)

    fig.suptitle(site)


# for each site collapses across cells and probes
fig, axes = plt.subplots(len(sites),
                         len(list(itt.combinations(meta['transitions'], 2))),
                         sharex=True, sharey=True)
for row, site in enumerate(sites):
    color = 'red' if site[0:3] == 'AMT' else 'black'
    collapsed = np.sum(sig_array[all_sites == site, :, :, :], axis=(0,1))
    for col, (ax, hist, pair) in enumerate(
            zip(axes[row], collapsed, itt.combinations(meta['transitions'], 2))):
        ax.bar(np.arange(collapsed.shape[-1]), hist, align='edge', color=color)

        if row == 0:
            ax.set_title(f'{pair[0]}_{pair[1]}')

        if col ==0:
            ax.set_ylabel(site)


# for each site collapses across cells, probes, and transition pairs
fig, axes = plt.subplots(len(sites), 1,
                         sharex=True, sharey=True)
for row, (site, ax) in enumerate(zip(sites, axes)):
    color = 'red' if site[0:3] == 'AMT' else 'black'
    collapsed = np.sum(sig_array[all_sites == site, :, :, :], axis=(0,1,2))

    ax.bar(np.arange(collapsed.shape[-1]), collapsed, align='edge', color=color)

    ax.set_ylabel(site)


# compares transitions: collapse across PEG vs A1, probe, cell
fig, axes = plt.subplots(2, 6, sharex=True, sharey=True)

for row, region in enumerate(['ley', 'AMT']):
    color = 'red' if region == 'AMT' else 'black'
    collapsed = np.mean(sig_array[all_regions == region, :, :, :], axis=(0,1))

    for col, (ax, hist, pair) in enumerate(
            zip(axes[row], collapsed, itt.combinations(meta['transitions'], 2))):
        ax.bar(np.arange(collapsed.shape[-1]), hist, align='edge', color=color)

        if row == 0:
            ax.set_title(f'{pair[0]}_{pair[1]}')

        if col ==0:
            ax.set_ylabel(region)


# plots all steps of analysis for this cell

def check_plot(cell, probe, tran_pair, significance=0.05):
    site = cell[:7]

    # loads the raw data
    recs = load(site)
    sig = recs['trip0']['resp']
    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()
    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = cdPCA.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                   smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                   zscore=meta['zscore'], part='probe')
    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    trialR = trialR.squeeze()


    cell_idx = goodcells.index(cell)
    prb_idx = all_probes.index(probe)
    t0_idx = meta['transitions'].index(tran_pair[0])
    t1_idx = meta['transitions'].index(tran_pair[1])
    pair_idx = trans_pairs.index(f'{tran_pair[0]}_{tran_pair[1]}')

    fig, axes = plt.subplots(4, 1, sharex=True)
    axes = np.ravel(axes)
    times = np.arange(trialR.shape[-1])

    # plots the PSTH and rasters of the compared responses
    axes[0].plot(trialR[:, cell_idx, t0_idx,:].mean(axis=0), color=trans_color_map[tran_pair[0]])
    axes[0].plot(trialR[:, cell_idx, t1_idx,:].mean(axis=0), color=trans_color_map[tran_pair[1]])

    bottom, top = axes[0].get_ylim()
    half = (top - bottom) / 2
    _ = fplt._raster(times, trialR[:, cell_idx, t0_idx,:], y_offset=0, y_range=(bottom, half), ax=axes[0],
                     scatter_kws={'color': trans_color_map[tran_pair[0]], 'alpha':0.8, 's':10})
    _ = fplt._raster(times, trialR[:, cell_idx, t1_idx,:], y_offset=0, y_range=(half, top), ax=axes[0],
                     scatter_kws={'color': trans_color_map[tran_pair[1]], 'alpha':0.8, 's':10})


    # plots the real dprime and the shuffled dprime
    axes[1].plot(all_reals[cell][prb_idx, pair_idx, :], color='black')
    # axes[1].plot(all_shuffled[cell][:, prb_idx, pair_idx, :].T, color='green', alpha=0.01)
    _ = fplt._cint(times, all_shuffled[cell][:, prb_idx, pair_idx,:], confidence=0.95, ax= axes[1],
                   fillkwargs={'color':'black', 'alpha':0.5})

    # plots the the calculated pvalue plust the significant threshold
    axes[2].plot(all_pvalues[cell][prb_idx, pair_idx, :],  color='black')
    axes[2].axhline(significance, color='red', linestyle='--')

    # plots the histogram of significant bins
    axes[3].bar(times, all_signif[cell][prb_idx, pair_idx, :], align='edge')
    return axes

cell = 'AMT028b-20-1'

axes = check_plot(cell, probe=6, tran_pair=('silence', 'continuous'), significance=threshold)




# organizes data in a pandas dataframe
'''
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
             #'transformation': transformation,
             'montecarlo': montecarlo,
             'pair': f'{trans_pair[0]}_{trans_pair[1]}',
             'threshold': pval_threshold,
             'parameter': 'total_sig',
             'value': total_sig}

        df.append(d)

        d = {'site': site,
             'probe': probe,
             #'transformation': transformation,
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
'''

'''
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
# fig.savefig(png, transparent=True, dpi=100)
svg = png = root.joinpath(f'paired_distance_summary').with_suffix('.svg')
# fig.savefig(svg, transparent=True)

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
'''
