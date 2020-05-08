import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl

import cpn_triplets as tp
from cpn_load import load
from reliability import signal_reliability

import cpn_dPCA as cdPCA

from scipy.stats import gaussian_kde as gkde
import seaborn as sn
import fancy_plots as plots

import pandas as pd
import seaborn as sn
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.stats import ranksums


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',
                  '#984ea3', '#999999', '#e41a1c', '#dede00']

# meta parameter
meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 20, # ms
        'raster_fs': 100,
        'transitions' : ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot' : [2,3,5,6],
        'significance': False}



code_to_name = {'t': 'Probe', 'ct': 'Context'}

all_sites = ['ley070a', # good site. A1
             'ley072b', # Primary looking responses with strong contextual effects
             'AMT028b', # good site
             'AMT029a', # Strong response, somehow visible contextual effects
             'AMT030a', # low responses, Ok but not as good
             #'AMT031a', # low response, bad
             'AMT032a'] # great site. PEG

df = list()

for site in all_sites:
    # load and format triplets from a site
    # site = 'AMT030a' # low responses, Ok but not as good
    recs = load(site)
    rec = recs['trip0']
    sig = rec['resp'].rasterize()

    # calculates response realiability and select only good cells to improve analysis

    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    if len(goodcells) < 10:
        n_components = len(goodcells)
    elif len(goodcells) == 0:
        pass#continue
    else:
        n_components = 10
    # plots PSTHs of all probes after silence
    # fig, axes = cplot.hybrid(sig, epoch_names=r'\AC0_P[2356]\Z', channels=goodcells)

    # plots PSHTs of individual best probe after all contexts
    # fig, axes = cplot.hybrid(sig, epoch_names=r'\AC\d_P3\Z', channels=goodcells)

    # takes an example probe
    full_array, invalid_cp, valid_cp, all_contexts, all_probes = \
        tp.make_full_array(sig, channels=goodcells, smooth_window=meta['smoothing_window'])


    for probe in meta['probes_to_plot']:

        # get a specific probe after a set of different transitions
        Z, trialZ, significance_masks, dpca = cdPCA.tran_dpca(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                                     smooth_window=meta['smoothing_window'], significance=meta['significance'],
                                                     raster_fs=meta['raster_fs'])
        expl_var = dpca.explained_variance_ratio_
        # plots the first PC projection of each context, for each marginalization
        # includes a measurement of significance by shuffle test

        for marg, vals  in expl_var.items():

            value = np.sum(vals) * 100

            d = {'site': site,
                 'probe': probe,
                 'marg': marg,
                 'value': value}

            df.append(d)

        # context dependent to context independent ratio
        value = np.sum(expl_var['ct']) / np.sum(expl_var['t'])

        d = {'site': site,
             'probe': probe,
             'marg': 'ratio',
             'value': value}
        df.append(d)



DF = pd.DataFrame(df)
# [y if y not in b else other_value for y in a]
DF['area'] = ['A1' if site[0:3] == 'ley' else 'PEG' for site in DF.site]
DF['unique'] = [f'{site}_P{probe}' for site, probe in zip(DF.site, DF.probe)]

fig, ax = plt.subplots()
for area, color in zip(DF.area.unique(), ['gray', 'green']):
    filtered = DF.loc[DF.area == area, :]
    pivoted = filtered.pivot(index='unique', columns='marg', values='value')
    ax.scatter(pivoted.t, pivoted.ct, color=color, label=area)

inset = inset_axes(ax, width='30%', height='50%')

ratios = DF.loc[DF.marg == 'ratio', : ]

pallette = sn.set_palette(['gray', 'green'])
inset = sn.swarmplot(x='area',y='value',  data=ratios, palette=pallette, dodge=True, ax=inset)
inset.set_ylabel('context/probe\nexpalinede variance ratio')

A1 = ratios.loc[ratios.area == 'A1', 'value'].values
PEG = ratios.loc[ratios.area == 'PEG', 'value'].values

stat, pval = ranksums(A1, PEG)

ax.set_xlabel('probe explained variance (%)')
ax.set_ylabel('context explained variance (%)')

inset.set_ylabel('context/probe\nexpalinede variance ratio')
inset.set_title('ranksums {:.3f}'.format(pval))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=15)
ax.title.set_size(20)
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)

fig.set_size_inches([5.82, 5.68])

root = pl.Path(f'/home/mateo/Pictures/DAC2')
if not root.exists(): root.mkdir(parents=True, exist_ok=True)
png = root.joinpath(f'dPCA_Summary').with_suffix('.png')
fig.savefig(png, transparent=True, dpi=100)
svg = png = root.joinpath(f'dPCA_Summary').with_suffix('.svg')
fig.savefig(svg, transparent=True)








