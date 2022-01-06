import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jl

from src.root_path import config_path

from src.data.rasters import load_site_formated_raster
from src.metrics.consolidated_dprimes import single_cell_dprimes
from src.metrics.significance import _significance


"""
single cell analysis: comparison between regions and context types i.e. silence, same, different.
"""


plt.style.use(['default', config_path / 'presentation.mplstyle'])

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None,
        'stim_type': 'permutations',
        'alpha':0.05}
# todo, if batch analysis rerun, use the anotated line instead
# summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'211221_cxt_metrics_summary_DF_alpha_{meta}'
summary_DF_file = pl.Path(config['paths']['analysis_cache']) / '211221_cxt_metrics_summary_DF_alpha_0.05'

DF = jl.load(summary_DF_file)

def format_dataframe(DF):

    ff_analylis = DF.analysis.isin(['SC', 'fdPCA'])
    ff_corr = DF.mult_comp_corr == 'consecutive_3'

    good_cols =['analysis', 'mult_comp_corr', 'region', 'siteid',  'cellid', 'context_pair',
                'probe', 'metric', 'value']
    filtered = DF.loc[ff_analylis & ff_corr, good_cols]

    filtered['probe'] = [int(p) for p in filtered['probe']]
    filtered['context_pair'] = [f"{int(cp.split('_')[0]):02d}_{int(cp.split('_')[1]):02d}"
                                for cp in filtered['context_pair']]

    # rename metrics and analysis for ease of ploting
    filtered['metric'] = filtered['metric'].replace({'significant_abs_mass_center': 'center of mass (ms)',
                                                     'significant_abs_mean': "mean d'",
                                                     'significant_abs_sum': "integral (d'*ms)"})
    filtered['analysis'] = filtered['analysis'].replace({'SC': 'single cell',
                                                         'fdPCA': 'population',
                                                         'pdPCA': 'probewise pop',
                                                         'LDA': 'pop ceiling'})

    filtered['id'] = filtered['cellid'].fillna(value=filtered['siteid'])
    filtered = filtered.drop(columns=['cellid', 'siteid'])

    filtered['value'] = filtered['value'].fillna(value=0)

    # permutation related preprocesing.
    # creates a new column relating probe with  context pairs
    ctx = np.asarray([row.split('_') for row in filtered.context_pair], dtype=int)
    prb = np.asarray(filtered.probe, dtype=int)

    silence = ctx == 0
    same = ctx == prb[:,None]
    different = np.logical_and(~silence, ~same)

    name_arr = np.full_like(ctx, np.nan, dtype=object)
    name_arr[silence] = 'silence'
    name_arr[same] = 'same'
    name_arr[different] = 'diff'
    comp_name_arr = np.apply_along_axis('_'.join, 1, name_arr)

    # swaps clasification names to not have repetitions i.e. diff_same == same_diff
    comp_name_arr[np.where(comp_name_arr == 'same_silence')] = 'silence_same'
    comp_name_arr[np.where(comp_name_arr == 'diff_silence')] = 'silence_diff'
    comp_name_arr[np.where(comp_name_arr == 'diff_same')] = 'same_diff'
    comp_name_arr[np.where(comp_name_arr == 'same_silence')] = 'silence_same'

    filtered['trans_pair'] = comp_name_arr

    ord_cols = ['analysis', 'region', 'id', 'context_pair', 'trans_pair', 'probe', 'metric', 'value']
    pivot_idx = [col for col in ord_cols if col not in ['value', 'metric']]
    pivoted = filtered.pivot_table(index=pivot_idx, columns='metric', values='value', aggfunc='first').reset_index()

    full_long = filtered # saves long format for subsamplig analysis

    return pivoted, full_long
pivoted, filtered = format_dataframe(DF)


##############################################################################
########## figure and subplots locations #####################################
##############################################################################

gs_kw = dict(width_ratios=[1, 0.5], height_ratios=[1, 1])
fig, axes = plt.subplots(2, 2, sharex='col', sharey='row', gridspec_kw=gs_kw)
fig.set_size_inches(10, 10)

##############################################################################
########## block of simple metric comparisons#################################
##############################################################################
trans_ord = ['diff_diff', 'same_diff', 'silence_same', 'silence_diff']
reg_ord = ['A1', 'PEG']

metrics = ["integral (d'*ms)", "center of mass (ms)"]
for nn, (metric, row) in enumerate(zip(metrics, axes)):
    tocalc = filtered.loc[(filtered.metric == metric) &
                          (filtered.analysis == 'single cell') &
                          (filtered.value >0),:]

    toplot = tocalc.loc[(tocalc.value < 600), :]

    if metric == "integral (d'*ms)":
        mname = 'integral'
        ylname = "discrimnation magnitude\n(d' * ms)"
        lab_color = 'C2'
    elif metric == "center of mass (ms)":
        mname = 'center_of_mass'
        ylname = "discrimination duration\n(ms)"
        lab_color = 'C4'

    trans_ax, reg_ax = row

    #transitions
    _ = sns.pointplot(x='trans_pair', y='value', data=tocalc, order=trans_ord, hue_order=reg_ord,
                       s=2, dodge=0.4, join=False, palette=['black'], capsize=0.2, ci=68, ax=trans_ax)
    trans_ax.legend([],[], frameon=False)
    trans_ax.set_xticklabels(trans_ord, rotation = 45)

    if nn>0: trans_ax.set_xlabel('context transition pair')
    trans_ax.set_ylabel(ylname)
    trans_ax.yaxis.label.set_color(lab_color)

    # regions
    _ = sns.pointplot(x='region', y='value', data=tocalc, order=reg_ord, hue_order=reg_ord,
                       s=2, dodge=0.4, join=False, palette=sns.color_palette(), capsize=0.2, ci=68, ax=reg_ax)

    reg_ax.legend([],[], frameon=False)
    reg_ax.set_xticklabels(reg_ord, rotation = 45)
    if nn>0: reg_ax.set_xlabel('region')
    reg_ax.set_ylabel("")

    fig.align_labels()






