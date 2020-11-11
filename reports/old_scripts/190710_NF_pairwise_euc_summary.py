import matplotlib.pyplot as plt
import numpy as np
import itertools as itt

from src.data.load import load
from src.metrics.reliability import signal_reliability
from src.metrics import trp_dispersion as ndisp
from src.data.cache import make_cache, get_cache
from src.data import rasters as tp

import pandas as pd

import seaborn as sn

from scipy.stats import ranksums, wilcoxon


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',
                  '#984ea3', '#999999', '#e41a1c', '#dede00']

meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 20, # ms
        'transitions' : ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot' : [2,3,5,6],
        'significance': False,
        'nonparam_shuffle': 1000}
fs = 100 #FixMe put into meta, chaches will need to be regenerated


all_sites = ['ley070a', # good site. A1
             'ley072b', # Primary looking responses with strong contextual effects
             'AMT028b', # good site
             'AMT029a', # Strong response, somehow visible contextual effects
             'AMT030a', # low responses, Ok but not as good
             #'AMT031a', # low response, bad
             'AMT032a'] # great site. PEG


analysis_name = 'trans_euc-dist_pop-scram'


analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])


code_to_name = {'t': 'Probe', 'ct': 'Context'}

df = list()

for site in all_sites:

    # load and format triplets from a site
    recs = load(site)

    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
    sig = rec['resp'].rasterize()

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # gets the full raster
    full_array, invalid_cp, valid_cp, all_contexts, all_probes = \
        tp.make_full_array(sig, channels=goodcells, smooth_window=meta['smoothing_window'])

    full_array = full_array[..., 100:]

    for probe in meta['probes_to_plot']:

        trans_arr = tp._extract_triplets_sub_arr(probes=probe, context_types=meta['transitions'], full_array=full_array,
                                                 context_names=all_contexts, probe_names=all_probes, squeeze=False)



        # calculates the old single trial pairwise normalized euclidean distance
        for trans_pair in (itt.combinations(meta['transitions'], 2)):

                signal_name = f'190709_{site}_P{probe}_ct-marg'

                func_args = {'transitions_array': trans_arr, 'probe_names': [probe], 'context_transitions': trans_pair,
                             'probe_order': [probe], 'trans_order': meta['transitions'],
                             'shuffle_num': meta['nonparam_shuffle'], 'trial_combinations': True}

                shuffled_dispersion_time = make_cache(function=ndisp.transition_pair_comparison_by_trials,
                                                      func_args=func_args,
                                                      classobj_name=signal_name, recache=False,
                                                      cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}')
                real, trans_shuffle, trial_shuffle = get_cache(shuffled_dispersion_time)

                # pvalue = (msd_floor > obs_msd).sum() / shuffle_n
                pvalues = np.sum((real < trans_shuffle), axis=0) / meta['nonparam_shuffle']

                for pval_threshold in [0.05, 0.01, 0.001]:
                    signif = pvalues < pval_threshold
                    total_sig = np.sum(signif)
                    try:
                        last_sig = np.max(np.argwhere(signif))
                    except: # if there is not significant
                        last_sig = 0

                    d = {'site': site,
                         'probe': probe,
                         'pair': f'{trans_pair[0]}_{trans_pair[1]}',
                         'threshold': pval_threshold,
                         'parameter': 'total_sig',
                         'value': total_sig}

                    df.append(d)

                    d = {'site': site,
                         'probe': probe,
                         'pair': f'{trans_pair[0]}_{trans_pair[1]}',
                         'threshold': pval_threshold,
                         'parameter': 'last_sig',
                         'value': last_sig}
                    df.append(d)


DF = pd.DataFrame(df)
DF['area'] = ['A1' if site[0:3] == 'ley' else 'PEG' for site in DF.site]
DF['unique'] = [f'{site}_P{probe}' for site, probe in zip(DF.site, DF.probe)]

ff_thres = DF.threshold == 0.01
ff_total = DF.parameter == 'total_sig'
ff_last = DF.parameter == 'last_sig'

filtered = DF.loc[ff_thres, :]
pallette = sn.set_palette(['gray', 'green'])
fig, ax  = plt.subplots()
ax = sn.swarmplot(x='parameter', y='value', hue='area', data=filtered, palette=pallette, dodge=True)

A1 = DF.area == 'A1'
PEG = DF.area == 'PEG'


total_A1 = DF.loc[ff_thres & A1 & ff_total, 'value'].values
total_PEG = DF.loc[ff_thres & PEG & ff_total, 'value'].values

last_A1 = DF.loc[ff_thres & A1 & ff_last, 'value'].values
last_PEG = DF.loc[ff_thres & PEG & ff_last, 'value'].values

_, tot_pval = ranksums(total_A1, total_PEG)
_, las_pval = ranksums(last_A1, last_PEG)

text = f'total_sig pval: {tot_pval}\nlast_sig pval: {las_pval}'

ax.text(-0.45, 60, text)

# set figure to full size in tenrec screen
fig.set_size_inches(19.2, 9.79)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=15)
ax.title.set_size(20)
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)

fig.set_size_inches([5.82, 5.68])

# root = pl.Path(f'/home/mateo/Pictures/DAC2')
# if not root.exists(): root.mkdir(parents=True, exist_ok=True)
# png = root.joinpath(f'paired_distance_summary').with_suffix('.png')
# fig.savefig(png, transparent=True, dpi=100)
# svg = png = root.joinpath(f'paired_distance_summary').with_suffix('.svg')
# fig.savefig(svg, transparent=True)

###
# now plots for PEG, comparisons between context pairs

for thres in [0.01, 0.05]:
    ff_thres = DF.threshold == thres
    ff_param = DF.parameter == 'total_sig'
    ff_area = DF.area == 'PEG'

    filtered = DF.loc[ff_thres & ff_param & ff_area, :]
    fig, ax  = plt.subplots()
    ax = sn.swarmplot(x='pair', y='value', data=filtered, color='green', dodge=True)


    # get all the pairwise p values
    # only for PEG

    pair_comparison = dict()
    for p1, p2 in itt.combinations(DF.pair.unique(), 2):

        ff_pairs = DF.pair.isin((p1,p2))
        ff_param = DF.parameter == 'total_sig'
        filtered = DF.loc[ff_thres & ff_pairs & ff_area & ff_param, :]
        pivoted = filtered.pivot(index='unique', columns='pair', values='value')

        x = pivoted[p1].values
        y = pivoted[p2].values

        _, pval = wilcoxon(x, y)
        pair_comparison[f'{p1} vs {p2}'] = pval


    text = [f'{key}: {val}' for key, val in pair_comparison.items()]
    text = '\n'.join(text)

    ax.text(-0.45, 40, text)

    # set figure to full size in tenrec screen
    fig.set_size_inches(19.2, 9.79)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(labelsize=15)
    ax.title.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)

    # tname =  str(thres).replace('.', ',')
    # root = pl.Path(f'/home/mateo/Pictures/DAC2')
    # if not root.exists(): root.mkdir(parents=True, exist_ok=True)
    # png = root.joinpath(f'paired_distance_p<{tname}').with_suffix('.png')
    # fig.savefig(png, transparent=True, dpi=100)
    # svg = png = root.joinpath(f'paired_distance_p<{tname}').with_suffix('.svg')
    # fig.savefig(svg, transparent=True)










