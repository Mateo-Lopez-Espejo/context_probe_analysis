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

import fancy_plots as fplt
from cpp_cache import set_name

"""
2020-05-??
Used an exponential decay to model the volution of contextual effectsos over time. Here thee fitted parameters (tau and 
y intercept r0) are compared across different treatments (probes, transitions_pairs), between single cell and population
analysis (dPCA, LDA) and finally between fitting the dprime or its profile of significance. 

tau is selected from the fitted significance profile, and r0 form the fitted dprime
"""

config = ConfigParser()
if pl.Path('../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../context_probe_analysis/config/settings.ini'))
elif pl.Path('../../../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../../../context_probe_analysis/config/settings.ini'))
else:
    raise FileNotFoundError('config file coluld not be foud')

# analysis should be createde and cached with cpn_batch_dprime.py beforehand, using the same meta parameters
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
# compare tau between different probe means
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe != 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
ff_outliers = DF.value < 1000

filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
                  ['cellid', 'probe', 'value']]
pivoted = filtered.pivot(index='cellid', columns='probe', values='value').dropna().reset_index()
molten = pivoted.melt(id_vars='cellid', var_name='probe')

fig, ax = plt.subplots()
# ax = sns.violinplot(x='probe', y='value', data=molten, ax=ax, color='gray', cut=0)
ax = sns.swarmplot(x='probe', y='value', data=molten, ax=ax, color='gray')
sns.despine(ax=ax)

# no significant comparisons
box_pairs = list(itt.combinations(filtered.probe.unique(), 2))
# box_pairs = [('probe_2', 'probe_3'), ('probe_3', 'probe_5')]
stat_resutls = add_stat_annotation(ax, data=molten, x='probe', y='value', test='Wilcoxon',
                                   box_pairs=box_pairs, comparisons_correction=None)

ax.set_ylabel(f'tau (ms)', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)
ax.set_xlabel('', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'summary significance-tau comparison between probes'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'wip3_figures', title)
########################################################################################################################
# compare tau between different transition pair means
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair != 'mean'
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
ff_outliers = DF.value < 1000

filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
                  ['cellid', 'transition_pair', 'value']]
pivoted = filtered.pivot(index='cellid', columns='transition_pair', values='value').dropna().reset_index()
molten = pivoted.melt(id_vars='cellid', var_name='transition_pair')

fig, ax = plt.subplots()
# ax = sns.violinplot(x='transition_pair', y='value', data=molten, ax=ax, color='gray', cut=0)
ax = sns.swarmplot(x='transition_pair', y='value', data=molten, ax=ax, color='gray')
sns.despine(ax=ax)

# box_pairs = list(itt.combinations(filtered.transition_pair.unique(), 2))
box_pairs = [('continuous_sharp', 'continuous_similar'), ('continuous_similar', 'silence_continuous'),
             ('continuous_similar', 'silence_sharp'), ('continuous_similar', 'silence_similar'),
             ('continuous_similar', 'similar_sharp')]
stat_resutls = add_stat_annotation(ax, data=molten, x='transition_pair', y='value', test='Wilcoxon',
                                   box_pairs=box_pairs, comparisons_correction=None)

ax.set_ylabel(f'tau (ms)', fontsize=ax_lab_size)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.tick_params(labelsize=ax_val_size)
ax.set_xlabel('', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'summary significance-tau comparison between transitions'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'wip3_figures', title)

########################################################################################################################
# compare r0 between different probe means
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe != 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'

filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                  ['cellid', 'probe', 'value']]
pivoted = filtered.pivot(index='cellid', columns='probe', values='value').dropna().reset_index()
molten = pivoted.melt(id_vars='cellid', var_name='probe')

fig, ax = plt.subplots()
# ax = sns.violinplot(x='probe', y='value', data=molten, ax=ax, color='gray', cut=0)
ax = sns.swarmplot(x='probe', y='value', data=molten, ax=ax, color='gray')
sns.despine(ax=ax)

box_pairs = list(itt.combinations(filtered.probe.unique(), 2))
# box_pairs = [('probe_2', 'probe_3')]
stat_resutls = add_stat_annotation(ax, data=molten, x='probe', y='value', test='Wilcoxon',
                                   box_pairs=box_pairs, comparisons_correction=None)

ax.set_ylabel(f'amplitude (z-score)', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)
ax.set_xlabel('', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'summary dprime-r0 comparison between probes'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'wip3_figures', title)

########################################################################################################################
# compare r0 between different transition pair means
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair != 'mean'
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'

filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                  ['cellid', 'transition_pair', 'value']]
pivoted = filtered.pivot(index='cellid', columns='transition_pair', values='value').dropna().reset_index()
molten = pivoted.melt(id_vars='cellid', var_name='transition_pair')

fig, ax = plt.subplots()
# ax = sns.violinplot(x='transition_pair', y='value', data=molten, ax=ax, color='gray', cut=0)
ax = sns.swarmplot(x='transition_pair', y='value', data=molten, ax=ax, color='gray')
sns.despine(ax=ax)

box_pairs = list(itt.combinations(filtered.transition_pair.unique(), 2))
# box_pairs = [('continuous_sharp', 'continuous_similar'), ('continuous_sharp', 'silence_continuous'),
#              ('continuous_sharp', 'silence_sharp'), ('continuous_sharp', 'silence_similar'),
#              ('continuous_similar', 'silence_continuous'), ('continuous_similar', 'silence_sharp'),
#              ('continuous_similar', 'silence_similar'), ('continuous_similar', 'similar_sharp'),
#              ('silence_similar', 'similar_sharp')]
stat_resutls = add_stat_annotation(ax, data=molten, x='transition_pair', y='value', test='Wilcoxon',
                                   box_pairs=box_pairs, comparisons_correction=None)

ax.set_ylabel(f'amplitude (z-score)', fontsize=ax_lab_size)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.tick_params(labelsize=ax_val_size)
ax.set_xlabel('', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'summary dprime-r0 comparison between transitions'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'wip3_figures', title)

########################################################################################################################
# Distribution of cells in r0 tau space
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'
R0 = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
            ['region', 'siteid', 'cellid', 'parameter', 'value']]

ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
ff_outliers = DF.value < 2000
Tau = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
             ['region', 'siteid', 'cellid', 'parameter', 'value']]

filtered = pd.concat([R0, Tau])
pivoted = filtered.pivot_table(index=['region', 'siteid', 'cellid'],
                               columns='parameter', values='value').dropna().reset_index()

fig, ax = plt.subplots()
# ax = sns.scatterplot(x='r0', y='tau', data=pivoted, color='black')
ax = sns.regplot(x='r0', y='tau', data=pivoted, color='black')
sns.despine(ax=ax)

ax.set_ylabel(f'tau (ms)', fontsize=ax_lab_size)
ax.set_xlabel('amplitude (z-score)', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)

_, _, r2, _, _ = sst.linregress(pivoted.r0, pivoted.tau)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'all cell summary parameter space r={r2:.3f}'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'wip3_figures', title)

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
fplt.savefig(fig, 'wip3_figures', title)

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
fplt.savefig(fig, 'wip3_figures', title)

########################################################################################################################
# single cell comparison between regions and parameters
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'
R0 = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
            ['region', 'siteid', 'cellid', 'parameter', 'value']]

ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
ff_outliers = DF.value < 2000
Tau = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
             ['region', 'siteid', 'cellid', 'parameter', 'value']]

filtered = pd.concat([R0, Tau])
# molten = pivoted.melt(id_vars='cellid', var_name='transition_pair')

g = sns.catplot(x='region', y='value', col='parameter', data=filtered, kind="violin", cut=0,
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
fplt.savefig(fig, 'wip3_figures', title)

########################################################################################################################
# Compares tau between dprime and significance
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter.isin(['tau', 'r0'])
ff_outliers = DF.value < 10000
filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_outliers,
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
fplt.savefig(fig, 'wip3_figures', title)

########################################################################################################################
# dCPA dPCA
########################################################################################################################
# dPCA compare tau between different probe means
ff_anal = DF.analysis == 'dPCA'
ff_probe = DF.probe != 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
ff_outliers = DF.value < 2000

filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
                  ['siteid', 'probe', 'value']]
pivoted = filtered.pivot(index='siteid', columns='probe', values='value').dropna().reset_index()
molten = pivoted.melt(id_vars='siteid', var_name='probe')

fig, ax = plt.subplots()
# ax = sns.violinplot(x='probe', y='value', data=molten, ax=ax, color='gray', cut=0)
ax = sns.swarmplot(x='probe', y='value', data=molten, ax=ax, color='gray')
sns.despine(ax=ax)

# box_pairs = list(itt.combinations(filtered.probe.unique(), 2))
# box_pairs = [('probe_2', 'probe_3'), ('probe_3', 'probe_5')]
# stat_resutls = add_stat_annotation(ax, data=molten, x='probe', y='value', test='Wilcoxon',
#                                    box_pairs=box_pairs, comparisons_correction=None)

ax.set_ylabel(f'tau (ms)', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)
ax.set_xlabel('', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'dPCA summary significance-tau comparison between probes'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'wip3_figures', title)
########################################################################################################################
# dPCA compare tau between different transition pair means
ff_anal = DF.analysis == 'dPCA'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair != 'mean'
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
ff_outliers = DF.value < 2000

filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
                  ['siteid', 'transition_pair', 'value']]
pivoted = filtered.pivot(index='siteid', columns='transition_pair', values='value').dropna().reset_index()
molten = pivoted.melt(id_vars='siteid', var_name='transition_pair')

fig, ax = plt.subplots()
# ax = sns.violinplot(x='transition_pair', y='value', data=molten, ax=ax, color='gray', cut=0)
ax = sns.swarmplot(x='transition_pair', y='value', data=molten, ax=ax, color='gray')
sns.despine(ax=ax)

# box_pairs = list(itt.combinations(filtered.transition_pair.unique(), 2))
box_pairs = [('continuous_sharp', 'continuous_similar'), ('continuous_similar', 'silence_continuous')]
stat_resutls = add_stat_annotation(ax, data=molten, x='transition_pair', y='value', test='Wilcoxon',
                                   box_pairs=box_pairs, comparisons_correction=None)

ax.set_ylabel(f'tau (ms)', fontsize=ax_lab_size)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.tick_params(labelsize=ax_val_size)
ax.set_xlabel('', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'dPCA summary significance-tau comparison between transitions'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'wip3_figures', title)

########################################################################################################################
# dPCA compare r0 between different probe means
ff_anal = DF.analysis == 'dPCA'
ff_probe = DF.probe != 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'

filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                  ['siteid', 'probe', 'value']]
pivoted = filtered.pivot(index='siteid', columns='probe', values='value').dropna().reset_index()
molten = pivoted.melt(id_vars='siteid', var_name='probe')

fig, ax = plt.subplots()
# ax = sns.violinplot(x='probe', y='value', data=molten, ax=ax, color='gray', cut=0)
ax = sns.swarmplot(x='probe', y='value', data=molten, ax=ax, color='gray')
sns.despine(ax=ax)

# box_pairs = list(itt.combinations(filtered.probe.unique(), 2))
box_pairs = [('probe_2', 'probe_3')]
stat_resutls = add_stat_annotation(ax, data=molten, x='probe', y='value', test='Wilcoxon',
                                   box_pairs=box_pairs, comparisons_correction=None)

ax.set_ylabel(f'amplitude (z-score)', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)
ax.set_xlabel('', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'dPCA summary dprime-r0 comparison between probes'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'wip3_figures', title)

########################################################################################################################
# dPCA compare r0 between different transition pair means
ff_anal = DF.analysis == 'dPCA'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair != 'mean'
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'

filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                  ['siteid', 'transition_pair', 'value']]
pivoted = filtered.pivot(index='siteid', columns='transition_pair', values='value').dropna().reset_index()
molten = pivoted.melt(id_vars='siteid', var_name='transition_pair')

fig, ax = plt.subplots()
# ax = sns.violinplot(x='transition_pair', y='value', data=molten, ax=ax, color='gray', cut=0)
ax = sns.swarmplot(x='transition_pair', y='value', data=molten, ax=ax, color='gray')
sns.despine(ax=ax)

# box_pairs = list(itt.combinations(filtered.transition_pair.unique(), 2))
box_pairs = [('continuous_sharp', 'continuous_similar')]
stat_resutls = add_stat_annotation(ax, data=molten, x='transition_pair', y='value', test='Wilcoxon',
                                   box_pairs=box_pairs, comparisons_correction=None)

ax.set_ylabel(f'amplitude (z-score)', fontsize=ax_lab_size)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.tick_params(labelsize=ax_val_size)
ax.set_xlabel('', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6, 6))
title = f'dPCA summary dprime-r0 comparison between transitions'
fig.suptitle(title)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'wip3_figures', title)
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
ff_outliers = DF.value < 2000
Tau = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
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
fplt.savefig(fig, 'wip3_figures', title)

########################################################################################################################
# SC vs dPCA taus, filtering SC with r0 of dPCA
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
ff_outliers = DF.value < 2000
sing = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
              ['region', 'siteid', 'cellid', 'parameter', 'value']]

sing_pivot = sing.pivot(index='siteid', columns='cellid', values='value')

sing_pivot['max'] = sing_pivot.mean(axis=1)

ff_anal = DF.analysis == 'dPCA'
pops = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
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
fplt.savefig(fig, 'wip3_figures', title)

########################################################################################################################
# SC vs dPCA r0, filtering SC with r0 of dPCA
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'
ff_outliers = DF.value < 2000
sing = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
              ['region', 'siteid', 'cellid', 'parameter', 'value']]

sing_pivot = sing.pivot(index='siteid', columns='cellid', values='value')

sing_pivot['max'] = sing_pivot.mean(axis=1)

ff_anal = DF.analysis == 'dPCA'
pops = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
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
fplt.savefig(fig, 'wip3_figures', title)

########################################################################################################################
# SC mean vs dPCA taus, tau outliers filtered

ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
ff_outliers = DF.value < 2000
sing = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
              ['region', 'siteid', 'cellid', 'parameter', 'value']]

sing_pivot = sing.pivot(index='siteid', columns='cellid', values='value')

sing_pivot['max'] = sing_pivot.mean(axis=1)

ff_anal = DF.analysis == 'dPCA'
pops = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
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
fplt.savefig(fig, 'DAC3_figures', title)

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
fplt.savefig(fig, 'DAC3_figures', title)
