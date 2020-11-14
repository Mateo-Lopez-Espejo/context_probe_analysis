from configparser import ConfigParser
import pathlib as pl
import joblib as jl
import itertools as itt

import pandas as pd
import scipy.stats as sst

import matplotlib.pyplot as plt
import seaborn as sns

from src.visualization import fancy_plots as fplt
from src.data.cache import set_name

"""
2020-06-30
Previously I used a fitted exponential decay to describe the evolution of contextual effects over time, however the fitting
in many cases was unadequate, adding artifactual outliers. An alterlative to this fitting approach is to instead consider
a fitless alternative, like the integral of the contextual effect. Here I explore two alternatives of this
1. total integral of the dprime time series 
2. integral of the dprime series only at time bins with significant dprime values
"""

config = ConfigParser()
config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))

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
# Comparisons between fited paramters and different integrals
ff_anal = DF.analysis == 'SC'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter != 'max'
ff_source = DF.source == 'dprime'
ff_good = DF.goodness > 0.01
ff_nan = pd.isnull(DF.goodness)

filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & (ff_good | ff_nan),
            ['region', 'siteid', 'cellid', 'parameter', 'goodness', 'value']]

pivoted = filtered.pivot_table(index=['region', 'siteid', 'cellid'],
                               columns='parameter', values='value').dropna().reset_index()

X = ['tau', 'r0', 'sum', 'significant_sum']
Y = ['sum', 'sum', 'significant_sum', 'significant_abs_sum']
for x, y in zip(X, Y):

    fig, ax = plt.subplots()
    # ax = sns.scatterplot(x='r0', y='tau', data=pivoted, color='black')
    ax = sns.regplot(x=x, y=y, data=pivoted, color='black')
    # _ = fplt.unit_line(ax)
    sns.despine(ax=ax)
    ax.xaxis.label.set_size(ax_lab_size)
    ax.yaxis.label.set_size(ax_lab_size)
    ax.tick_params(labelsize=ax_val_size)

    _, _, r2, _, _ = sst.linregress(pivoted[x], pivoted[y])

    fig = ax.figure
    fig.set_size_inches((6, 6))
    title = f'all cells summary {x} vs {y} r={r2:.3f}'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fplt.savefig(fig, 'SFN20_figures', title)

########################################################################################################################
# Comparisons between integrals and probe or transitions_pairs
analyses = ['SC', 'dPCA']
sources = ['dprime']
parameters = ['sum']
comparisons = ['probe', 'transition_pair']

good_thresh = -100

for analysis, source, parameter, compare in itt.product(analyses, sources, parameters, comparisons):

    # for single plot
    # analysis = 'SC'
    # source = 'dprime'
    # parameter = 'significant_integral'
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
    ff_nan = pd.isna(DF.goodness)

    if analysis == 'SC':
        index = 'cellid'
    elif analysis in ('dPCA', 'LDA'):
        index = 'siteid'


    # defines subset of good cells or sites based on goodness of fit for dprime
    good_filter = DF.loc[ff_anal & (DF.parameter == 'tau') & ff_source & ff_probe & ff_trans & ff_good,
                      [index, compare, 'goodness']]
    good_pivot = good_filter.pivot(index=index, columns=compare, values='goodness').dropna().reset_index()
    good_idx = good_pivot[index].unique()
    ff_goodidx = DF[index].isin(good_idx)


    filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_goodidx,
                      [index, compare, 'value']]
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

    ax.set_ylabel(f'integral', fontsize=ax_lab_size)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.tick_params(labelsize=ax_val_size)
    ax.set_xlabel('', fontsize=ax_lab_size)
    ax.tick_params(labelsize=ax_val_size)

    fig = ax.figure
    fig.set_size_inches((6, 6))
    title = f'{analysis} {source}-{parameter} between {compare}, goodness {good_thresh}'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fplt.savefig(fig, 'SFN20_figures', title)


########################################################################################################################
# Comparisons between ingegral and significant integral for the subsets of probes mean... ?? ToDo complete
analyses = ['SC', 'dPCA']
sources = ['dprime']
parameters = ['integral', 'significant_integral']
comparisons = ['probe', 'transition_pair']

good_thresh = 0.1

for analysis, source, compare in itt.product(analyses, sources,comparisons):

    # for single plot
    analysis = 'SC'
    source = 'dprime'
    parameter = 'tau'
    compare = 'transition_pair'

    if compare == 'probe':
        ff_probe = DF.probe != 'mean'
        ff_trans = DF.transition_pair == 'mean'
    elif compare == 'transition_pair':
        ff_probe = DF.probe == 'mean'
        ff_trans = DF.transition_pair != 'mean'

    ff_anal = DF.analysis == analysis
    ff_param = DF.parameter.isin(parameters)
    ff_source = DF.source == source

    if analysis == 'SC':
        index = 'cellid'
    elif analysis in ('dPCA', 'LDA'):
        index = 'siteid'

    filtered = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                      [index, compare, 'goodness', 'value', 'parameter']]
    pivoted = filtered.pivot(index=index, columns='parameter', values='value').dropna().reset_index()

    fig, ax = plt.subplots()
    ax = sns.scatterplot(x=compare, y='value', data=pivoted, ax=ax, color='gray', width=0.5)
    sns.despine(ax=ax)

    # no significant comparisons

    ax.set_ylabel(f'tau (ms)', fontsize=ax_lab_size)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.tick_params(labelsize=ax_val_size)
    ax.set_xlabel('', fontsize=ax_lab_size)
    ax.tick_params(labelsize=ax_val_size)

    fig = ax.figure
    fig.set_size_inches((6, 6))
    title = f'{analysis} {source}-{parameter} between {compare}'
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fplt.savefig(fig, 'SFN20_figures', title)




