import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import joblib as jl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sst
import seaborn as sns
from statannot import add_stat_annotation

from src.visualization import fancy_plots as fplt
from src.data.cache import set_name

"""
2020-05-??
Used an exponential decay to model the evolution of contextual effects over time. Here thee fitted parameters (tau and 
y intercept r0) are compared across different treatments (probes, transitions_pairs), between single cell and population
analysis (dPCA, LDA) and finally between fitting the dprime or its profile of significance. 

tau is selected from the fitted significance profile, and r0 form the fitted dprime

2020-06-30
further finer selection is done considering the goodness of the fit.
outlier values tend to correspond with poor fits
Also compares the R2 goodness of fit with the standard error of the fitted parameters
"""

config = ConfigParser()
config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))

# analysis should be createde and cached with trp_batch_dprime.py beforehand, using the same meta parameters
meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None}

# transferable plotting parameters
plt.rcParams['svg.fonttype'] = 'none'
sup_title_size = 30
sub_title_size = 20
ax_lab_size = 15
ax_val_size = 11
full_screen = [19.2, 9.83]
sns.set_style("ticks")

########################################################################################################################
########################################################################################################################
# data frame containing all the important summary data, i.e. exponential decay fits for dprime and significance, for
# all combinations of transition pairs, and probes,  for the means across probes, transitions pairs or for both, and
# for the single cell analysis or the dPCA projections
summary_DF_file = pl.Path(config['paths']['analysis_cache']) / 'DF_summary' / set_name(meta)
print('loading cached summary DataFrame')
DF = jl.load(summary_DF_file)

########################################################################################################################
# SC
########################################################################################################################
# compare parameters between different probes or transitions pairs

analyses = ['SC', 'dPCA']
sources = ['dprime', 'significance']
parameters = ['tau', 'r0']
comparisons = ['probe', 'transition_pair']

good_thresh = 0.1

for analysis, source, parameter, compare in itt.product(analyses, sources, parameters, comparisons):

    # # for single plot
    # analysis = 'SC'
    # source = 'dprime'
    # parameter = 'tau'
    # compare = 'transition_pair'

    if compare == 'probe':
        ff_probe = DF.probe != 'mean'
        ff_trans = DF.transition_pair == 'mean'
    elif compare == 'transition_pair':
        ff_probe = DF.probe == 'mean'
        ff_trans = DF.transition_pair != 'mean'

    ff_anal = DF.analysis == analysis
    ff_param = DF.parameter == parameter
    ff_source = DF.source == source
    ff_good = DF.goodness > good_thresh

    if analysis == 'SC':
        index = 'cellid'
    elif analysis in ('dPCA', 'LDA'):
        index = 'siteid'

    filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_good,
                      [index, compare, 'goodness', 'value']]
    pivoted = filtered.pivot(index=index, columns=compare, values='value').dropna().reset_index()
    molten = pivoted.melt(id_vars=index, var_name=compare)

    fig, ax = plt.subplots()
    _ = fplt.paired_comparisons(ax, data=molten,x=compare, y='value', color='gray', alpha=0.3)
    ax = sns.boxplot(x=compare, y='value', data=molten, ax=ax, color='gray', width=0.5)
    sns.despine(ax=ax)

    # no significant comparisons
    box_pairs = list(itt.combinations(filtered[compare].unique(), 2))
    stat_resutls = fplt.add_stat_annotation(ax, data=molten, x=compare, y='value', test='Wilcoxon',
                                       box_pairs=box_pairs, width=0.5, comparisons_correction=None)

    ax.set_ylabel(f'tau (ms)', fontsize=ax_lab_size)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.tick_params(labelsize=ax_val_size)
    ax.set_xlabel('', fontsize=ax_lab_size)
    ax.tick_params(labelsize=ax_val_size)

    fig = ax.figure
    fig.set_size_inches((6, 6))
    title = f'{analysis} {source}-{parameter} between {compare} goodness {good_thresh}'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fplt.savefig(fig, 'SFN20_figures', title)

########################################################################################################################
# Distribution of cells in r0 tau space.
good_thresh = 0.1
r0_source = 'dprime'
tau_source = 'dprime'

ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'r0'
ff_source = DF.source == r0_source
ff_good = DF.goodness > good_thresh
R0 = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_good,
            ['region', 'siteid', 'cellid', 'parameter', 'value']]

ff_param = DF.parameter == 'tau'
ff_source = DF.source == tau_source
Tau = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_good,
             ['region', 'siteid', 'cellid', 'parameter', 'value']]

filtered = pd.concat([R0, Tau])
pivoted = filtered.pivot_table(index=['region', 'siteid', 'cellid'],
                               columns='parameter', values='value').dropna().reset_index()

fig, ax = plt.subplots()
ax = sns.regplot(x='r0', y='tau', data=pivoted, color='black')
sns.despine(ax=ax)

# ax.set_ylabel(f'tau (ms)', fontsize=ax_lab_size)
# ax.set_xlabel('amplitude (z-score)', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)

_, _, r2, _, _ = sst.linregress(pivoted.r0, pivoted.tau)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'all cell summary param space {r0_source}_r0 {tau_source}_tau r={r2:.3f} goodness {good_thresh}'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'SFN20_figures', title)

#########################################################
# cells in parameter space colored by site
fig, ax = plt.subplots()
# ax = sns.scatterplot(x='r0', y='tau', data=pivoted, color='black')
ax = sns.scatterplot(x='r0', y='tau', hue='siteid', data=pivoted, legend='full')
ax.legend(loc='upper right', fontsize='large', markerscale=1, frameon=False)
sns.despine(ax=ax)

ax.set_ylabel(f'tau (ms)', fontsize=ax_lab_size)
ax.set_xlabel('amplitude (z-score)', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'cells in parameter space by site'
fig.suptitle(title, fontsize=sub_title_size)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'SFN20_figures', title)

#########################################################
# cells in parameter space colored by region
fig, ax = plt.subplots()
# ax = sns.scatterplot(x='r0', y='tau', data=pivoted, color='black')
ax = sns.scatterplot(x='r0', y='tau', hue='region', data=pivoted, legend='full')
ax.legend(loc='upper right', fontsize='large', markerscale=1, frameon=False)
sns.despine(ax=ax)

ax.set_ylabel(f'tau (ms)', fontsize=ax_lab_size)
ax.set_xlabel('amplitude (z-score)', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'cells in parameter space by region'
fig.suptitle(title, fontsize=sub_title_size)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'SFN20_figures', title)

########################################################################################################################
# single cell comparison between regions and parameters
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'
ff_good = DF.goodness > 0.01
R0 = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_good,
            ['region', 'siteid', 'cellid', 'parameter', 'value']]

ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
Tau = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_good,
             ['region', 'siteid', 'cellid', 'parameter', 'value']]

filtered = pd.concat([R0, Tau])
# molten = pivoted.melt(id_vars='cellid', var_name='transition_pair')

g = sns.catplot(x='region', y='value', col='parameter', data=filtered, kind="swarm",
                sharex=True, sharey=False)
sns.despine()

# add significnace
for ax, param in zip(np.ravel(g.axes), filtered.parameter.unique()):

    sub_filtered = filtered.loc[filtered.parameter == param, :]

    box_pairs = [('PEG', 'A1')]
    stat_resutls = add_stat_annotation(ax, data=sub_filtered, x='region', y='value', test='Mann-Whitney',
                                       box_pairs=box_pairs, comparisons_correction=None)

    if param == 'r0':
        param = 'z-score'
    elif param == 'tau':
        param = 'ms'
    ax.set_ylabel(f'{param}', fontsize=ax_lab_size)
    ax.tick_params(labelsize=ax_val_size)
    ax.set_xlabel('', fontsize=ax_lab_size)
    ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'SC parameter comparison between regions'
fig.suptitle(title, fontsize=sub_title_size)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'SFN20_figures', title)

########################################################################################################################
# Compares tau between dprime and significance
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter.isin(['tau', 'r0'])
ff_good = DF.goodness > 0.01
filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_good,
                  ['cellid', 'source', 'parameter', 'value']]

pivoted = filtered.pivot_table(index=['cellid', 'parameter'], columns='source', values='value').dropna().reset_index()

facet_grid = sns.lmplot(x='dprime', y='significance', col='parameter', data=pivoted,
                        sharex=False, sharey=False, scatter_kws={'color': 'black'}, line_kws={'color': 'black'})
# draws unit line, formats ax
for ax in np.ravel(facet_grid.axes):
    _ = fplt.unit_line(ax)
    ax.xaxis.label.set_size(ax_lab_size)
    ax.yaxis.label.set_size(ax_lab_size)
    ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((16, 8))
title = f'significance vs dprime fitted params comparison'
fig.suptitle(title, fontsize=20)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'SFN20_figures', title)

########################################################################################################################
# dCPA dPCA
########################################################################################################################
# dPCA comparison between regions and parameters
ff_anal = DF.analysis == 'dPCA'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'
R0 = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
            ['region', 'siteid', 'cellid', 'parameter', 'value']]

ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
ff_good = DF.goodness > 0.01
Tau = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_good,
             ['region', 'siteid', 'cellid', 'parameter', 'value']]

filtered = pd.concat([R0, Tau])

# g = sns.catplot(x='region', y='value', col='parameter', data=filtered,  kind="violin", cut=0,
#                 sharex=True, sharey=False)
g = sns.catplot(x='region', y='value', col='parameter', data=filtered, kind="swarm",
                sharex=True, sharey=False)
sns.despine()

# add significnace
for ax, param in zip(np.ravel(g.axes), filtered.parameter.unique()):

    sub_filtered = filtered.loc[filtered.parameter == param, :]

    box_pairs = [('PEG', 'A1')]
    stat_resutls = add_stat_annotation(ax, data=sub_filtered, x='region', y='value', test='Mann-Whitney',
                                       box_pairs=box_pairs, comparisons_correction=None)

    if param == 'r0':
        param = 'z-score'
    elif param == 'tau':
        param = 'ms'
    ax.set_ylabel(f'{param}', fontsize=ax_lab_size)
    ax.tick_params(labelsize=ax_val_size)
    ax.set_xlabel('', fontsize=ax_lab_size)
    ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'dPCA parameter comparison between regions'
fig.suptitle(title, fontsize=sub_title_size)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'SFN20_figures', title)

########################################################################################################################
# SC vs dPCA taus, filtering SC with r0 of dPCA
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
ff_good = DF.goodness > 0.01
sing = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_good,
              ['region', 'siteid', 'cellid', 'parameter', 'value']]

sing_pivot = sing.pivot(index='siteid', columns='cellid', values='value')

sing_pivot['max'] = sing_pivot.mean(axis=1)

ff_anal = DF.analysis == 'dPCA'
pops = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_good,
              ['region', 'siteid', 'cellid', 'parameter', 'value']]

pops = pops.set_index('siteid')

toplot = pd.concat((pops.loc[:, ['region', 'value']], sing_pivot.loc[:, 'max']), axis=1)

fig, ax = plt.subplots()
ax = sns.regplot(x='value', y='max', data=toplot, color='black', ax=ax)
sns.despine(ax=ax)
_, _, r2, _, _ = sst.linregress(toplot['value'], toplot['max'])
_ = fplt.unit_line(ax, square_shape=False)

ax.set_xlabel(f'dPCA tau (ms)', fontsize=ax_lab_size)
ax.set_ylabel(f'single cell mean tau (ms)', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = 'SC dPCA tau comparison r={:.2f}'.format(r2)
fig.suptitle(title, fontsize=sub_title_size)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'SFN20_figures', title)

########################################################################################################################
# SC vs dPCA r0, filtering SC with r0 of dPCA
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'
ff_good = DF.goodness > 0.01
sing = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_good,
              ['region', 'siteid', 'cellid', 'parameter', 'value']]

sing_pivot = sing.pivot(index='siteid', columns='cellid', values='value')

sing_pivot['max'] = sing_pivot.mean(axis=1)

ff_anal = DF.analysis == 'dPCA'
pops = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_good,
              ['region', 'siteid', 'cellid', 'parameter', 'value']]

pops = pops.set_index('siteid')

toplot = pd.concat((pops.loc[:, ['region', 'value']], sing_pivot.loc[:, 'max']), axis=1)

fig, ax = plt.subplots()
ax = sns.regplot(x='value', y='max', data=toplot, color='black', ax=ax)
sns.despine(ax=ax)
_, _, r2, _, _ = sst.linregress(toplot['value'], toplot['max'])
_ = fplt.unit_line(ax, square_shape=False)

ax.set_xlabel(f'dPCA amplitude (z-score)', fontsize=ax_lab_size)
ax.set_ylabel(f'single cell mean amplitude (z-score)', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'SC dPCA r0 comparison r={r2}'
fig.suptitle(title, fontsize=sub_title_size)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'SFN20_figures', title)

########################################################################################################################
# SC mean vs dPCA taus, tau outliers filtered

ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
ff_good = DF.goodness > 0.01
sing = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_good,
              ['region', 'siteid', 'cellid', 'parameter', 'value']]

sing_pivot = sing.pivot(index='siteid', columns='cellid', values='value')

sing_pivot['max'] = sing_pivot.mean(axis=1)

ff_anal = DF.analysis == 'dPCA'
pops = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_good,
              ['region', 'siteid', 'cellid', 'parameter', 'value']]

pops = pops.set_index('siteid')

toplot = pd.concat((pops.loc[:, ['region', 'value']], sing_pivot.loc[:, 'max']), axis=1)

fig, ax = plt.subplots()
ax = sns.regplot(x='value', y='max', data=toplot, color='black', ax=ax)
sns.despine(ax=ax)
_, _, r2, _, _ = sst.linregress(toplot['value'], toplot['max'])
_ = fplt.unit_line(ax, square_shape=False)

ax.set_xlabel(f'dPCA tau (ms)', fontsize=ax_lab_size)
ax.set_ylabel(f'single cell mean tau (ms)', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = 'SC mean dPCA tau comparison r={:.2f}'.format(r2)
fig.suptitle(title, fontsize=sub_title_size)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'SFN20_figures', title)

########################################################################################################################
# creates data frame with rows == cellid and columns == single cell dprime-r0, significance-tau, and their population
# equivalents.
# Plots all single cell values vs population value, eg. r0
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'

# single cell array
ff_anal = DF.analysis == 'SC'
# tau
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
SC_tau = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                ['region', 'siteid', 'cellid', 'value']]
SC_tau = SC_tau.set_index('cellid').rename(columns={'value': 'SC_tau'})
# r0
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'
SC_r0 = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
               ['region', 'siteid', 'cellid', 'value']]
SC_r0 = SC_r0.set_index('cellid').rename(columns={'value': 'SC_r0'})

# merge
SC_DF = pd.concat([SC_tau, SC_r0['SC_r0']], axis=1).copy()
SC_DF['dPCA_tau'] = np.nan
SC_DF['dPCA_r0'] = np.nan

# populatio values
ff_anal = DF.analysis == 'dPCA'
# tau
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
dPCA_tau = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                  ['region', 'siteid', 'value']]
dPCA_tau = dPCA_tau.set_index('siteid').rename(columns={'value': 'dPCA_tau'})

# r0
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'
dPCA_r0 = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                 ['region', 'siteid', 'value']]
dPCA_r0 = dPCA_r0.set_index('siteid').rename(columns={'value': 'dPCA_r0'})

# merge
dPCA_DF = pd.concat([dPCA_tau, dPCA_r0['dPCA_r0']], axis=1)

# apply population values to single SC_DF
for cellid, row in SC_DF.iterrows():
    site = row['siteid']
    SC_DF.loc[cellid, 'dPCA_tau'] = dPCA_DF.loc[site, 'dPCA_tau']
    SC_DF.loc[cellid, 'dPCA_r0'] = dPCA_DF.loc[site, 'dPCA_r0']

# filter out awnomalous data
ff_r0 = SC_DF['SC_r0'] >= 0.2
ff_tau = SC_DF['SC_tau'] <= 2000

toplot = SC_DF.loc[ff_tau & ff_r0, :]

x = 'dPCA_r0'
y = 'SC_r0'

fig, ax = plt.subplots()
ax = sns.regplot(x=x, y=y, data=toplot, color='black', ax=ax)
sns.despine(ax=ax)
left, right = ax.get_xlim()
ax.set_xlim(left, right + (right - left) / 12)
_, _, r2, _, _ = sst.linregress(toplot[x], toplot[y])
_ = fplt.unit_line(ax, square_shape=False)

ax.set_xlabel(x, fontsize=ax_lab_size)
ax.set_ylabel(y, fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = '{} vs {}, r={:.2f}'.format(x, y, r2)
fig.suptitle(title, fontsize=sub_title_size)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'SFN20_figures', title)

########################################################################################################################
########################################################################################################################
########################################################################################################################

# 2020-06-30 plots related to the new goodness of fit and parameter error values, both comparing them and using them to
# filter previoulsy done plots

########################################################################################################################
# common filtering for parameter, goodness and error comparisons
ff_anal = DF.analysis == 'SC'
ff_source = DF.source == 'dprime'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_goodness = DF.goodness > 0.1

filtered = DF.loc[ff_anal & ff_source & ff_probe & ff_trans & ff_goodness,
                  ['cellid', 'parameter', 'std', 'goodness', 'value']].dropna(axis='index')

###################################################################
# 1. compare fitted parameters to the goodness of fit.
#   it is notable that there an expected correlation, and filtering for goodness should lead to cleaner results
# 2. compare fitted paramter value to parameter error
#   unexpectedly the higher the fitted value, the higher the error,
# 3. compare the goodness of fit to the parameter error
#   there is a possitive correlation for r0 but not for tau. I would expect a negative correlation in both cases

X = ['value', 'value', 'std']
Y = ['goodness', 'std', 'goodness']

for x, y in zip(X, Y):
    g = sns.lmplot(x=x, y=y, data=filtered, col='parameter', fit_reg=True, sharex=False, sharey=False)
    axes = np.ravel(g.axes)
    fig = g.fig

    fig.set_size_inches((6, 6))
    title = f'SC {x} vs {y}'
    fig.suptitle(title, fontsize=sub_title_size)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fplt.savefig(fig, 'SFN20_figures', title)

########################################################################################################################
# creates data frame with rows == cellid and columns == single cell dprime-r0, significance-tau, and their population
# equivalents.
# Plots all single cell values vs population value, eg. r0
# filters based on goodness of fit over 0.01??
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'

# single cell array
ff_anal = DF.analysis == 'SC'
# tau
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'dprime'
SC_tau = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                ['region', 'siteid', 'cellid', 'goodness', 'value']]
SC_tau = SC_tau.set_index('cellid').rename(columns={'value': 'SC_tau'})
# r0
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'
SC_r0 = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
               ['region', 'siteid', 'cellid', 'goodness', 'value']]
SC_r0 = SC_r0.set_index('cellid').rename(columns={'value': 'SC_r0'})

# merge
SC_DF = pd.concat([SC_tau, SC_r0['SC_r0']], axis=1).rename(columns={'goodness': 'SC_goodness'}).copy()
# creates empty columns to later fill with population values
SC_DF['dPCA_tau'] = np.nan
SC_DF['dPCA_r0'] = np.nan
SC_DF['dPCA_goodness'] = np.nan

# population values
ff_anal = DF.analysis == 'dPCA'
# tau
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'dprime'
dPCA_tau = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                  ['region', 'siteid', 'goodness', 'value']]
dPCA_tau = dPCA_tau.set_index('siteid').rename(columns={'value': 'dPCA_tau'})

# r0
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'
dPCA_r0 = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                 ['region', 'siteid', 'goodness', 'value']]
dPCA_r0 = dPCA_r0.set_index('siteid').rename(columns={'value': 'dPCA_r0'})

# merge
dPCA_DF = pd.concat([dPCA_tau, dPCA_r0['dPCA_r0']], axis=1)

# apply population values to single SC_DF
for cellid, row in SC_DF.iterrows():
    site = row['siteid']
    SC_DF.loc[cellid, 'dPCA_tau'] = dPCA_DF.loc[site, 'dPCA_tau']
    SC_DF.loc[cellid, 'dPCA_r0'] = dPCA_DF.loc[site, 'dPCA_r0']
    SC_DF.loc[cellid, 'dPCA_goodness'] = dPCA_DF.loc[site, 'goodness']

# filter out anomalous data
# ff_r0 = SC_DF['SC_r0'] >= 0.2
# ff_tau = SC_DF['SC_tau'] <= 2000
ff_good = SC_DF['SC_goodness'] > 0.01

toplot = SC_DF.loc[ff_good, :]

x = 'dPCA_tau'
y = 'SC_tau'

fig, ax = plt.subplots()
ax = sns.regplot(x=x, y=y, data=toplot, color='black', ax=ax)
sns.despine(ax=ax)
left, right = ax.get_xlim()
ax.set_xlim(left, right + (right - left) / 12)
_, _, r2, _, _ = sst.linregress(toplot[x], toplot[y])
_ = fplt.unit_line(ax, square_shape=False)

ax.set_xlabel(x, fontsize=ax_lab_size)
ax.set_ylabel(y, fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = '{} vs {}, r={:.2f}'.format(x, y, r2)
fig.suptitle(title, fontsize=sub_title_size)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'SFN20_figures', title)

########################################################################################################################