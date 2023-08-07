import pathlib as pl
from configparser import ConfigParser

import joblib as jl
import numpy as np
import pandas as pd
import scipy.stats as sst
import scipy.signal as snl

from src.utils.dataframes import add_classified_contexts, norm_by_mean, \
    simplify_classified_contexts, add_ctx_type_voc
from src.root_path import config_path

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

# ToDo Start simplifying this mess of dataframes
###################### figure 2 ###################################


meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 20,
        'zscore': True,
        'stim_type': 'permutations'}
alpha = 0.05
montecarlos = 11000
summary_DF_file = pl.Path(
    config['paths']['analysis_cache']) / '220520_minimal_DF'
metrics = ['integral', 'last_bin']

DF_f2 = jl.load(summary_DF_file).query(
    f"source == 'real' and metric in {metrics} and "
    f"cluster_threshold == 0.05 and mult_comp_corr == 'bf_cp' and "
    f"analysis == 'SC' and "
    f"diff_metric == 'delta_FR' and "
    f"value > 0")

DF_f2.loc[
    DF_f2.metric == 'integral', 'value'
] = DF_f2.loc[
        DF_f2.metric == 'integral', 'value'
    ] / 1000  # ms to s for better display

DF_f2.drop(
    columns=['source', 'cluster_threshold', 'mult_comp_corr', 'analysis',
             'stim_count', ], inplace=True)
DF_f2.reset_index(drop=True, inplace=True)
DF_f2 = add_classified_contexts(DF_f2)

for col in ['id', 'context_pair', 'probe', 'site', 'region', 'metric',
            'trans_pair']:
    DF_f2[col] = DF_f2[col].astype('category')

DF_f2['value'] = pd.to_numeric(DF_f2.value, downcast='float')

# pivots for the scatter
pivoted_f2 = DF_f2.pivot_table(
    index=['region', 'site', 'context_pair', 'probe', 'trans_pair', 'id'],
    columns=['metric'], values='value', aggfunc='first',
    observed=True).reset_index()

# adds a small amount of jitter to the last bin value to help visualization
binsize = 1 / meta['raster_fs']
jitter = (np.random.random(pivoted_f2.shape[0]) * binsize * 0.8 - (
        binsize * 0.8 / 2)) * 1000  # in ms
pivoted_f2['last_bin_jittered'] = pivoted_f2['last_bin'] + jitter

# load and format ferret vocalization dataframe ToDo clean DFs
relevant_cols = ['id', 'region', 'named_ctx_0', 'named_ctx_1', 'named_probe',
                 'context_pair', 'probe', 'voc_ctx', 'voc_prb', 'metric',
                 'value']
ferret_df = pd.read_csv(
    pl.Path(config['paths']['analysis_cache']) / "230308_ferret_voc.csv",
    usecols=relevant_cols,
    dtype={key: ('category' if key != 'value' else 'float32') for key in
           relevant_cols})

fprobes = ferret_df['probe'].unique().tolist()
ferret_df['probe'] = ferret_df['probe'].replace(
    {p: int(p) for p in fprobes})  # replace category of str for cat of int

ferret_df = add_classified_contexts(ferret_df)

toregress_f2 = norm_by_mean(ferret_df)
toregress_f2 = simplify_classified_contexts(toregress_f2)

toplot_f2 = list()  # concatenate data classified on single values
for cat in ['diff', 'same', 'silence']:
    subset = toregress_f2.query(f"{cat} == 1").copy()
    subset['transition'] = cat
    toplot_f2.append(subset)

toplot_f2 = pd.concat(toplot_f2)

ferret_df = add_ctx_type_voc(ferret_df)

###################### figure 3 ###################################


mass_cluster_DF_file = pl.Path(
    config['paths']['analysis_cache']) / f'220520_minimal_DF'
DF_f3 = jl.load(mass_cluster_DF_file)

DF_f3.query("source == 'real' and mult_comp_corr in ['bf_cp', 'bf_ncp']  and "
            "metric in ['integral']", inplace=True)

DF_f3.loc[DF_f3.analysis == 'PCA', 'PC'] = DF_f3.loc[
    DF_f3.analysis == 'PCA', 'id'
].apply(lambda x: int(x.split('-')[-1]))

DF_f3.loc[
    DF_f3.metric == 'integral', 'value'
] = DF_f3.loc[
        DF_f3.metric == 'integral', 'value'
    ] / 1000

SC_DF = DF_f3.query("analysis == 'SC' and mult_comp_corr == 'bf_cp'"
                    " and metric == 'integral'")
PCA_DF = DF_f3.query("analysis == 'PCA' and mult_comp_corr == 'bf_cp'"
                     " and metric == 'integral'").copy()

# ToDo bring stuff over here and perhaps merge with DFs above

###################### figure 4 ###################################

# Baseline minimal parameters
raster_meta = {'reliability': 0.1,  # r value
               'smoothing_window': 0,  # ms
               'raster_fs': 20,
               'zscore': True,
               'stim_type': 'permutations'}

###################### figure 5 ###################################

# todo cleanup and organize these classifications to hold just the necesary

type_DF_file = pl.Path(
    config['paths']['analysis_cache']) / '220816_CPN_celltype_DF'
toclust_f5 = jl.load(type_DF_file).dropna(subset='sw')
toclust_f5['siteid'] = toclust_f5['id'].str.split('-').str[0]

# define kernel density estimate, the bandwidth is defined empirically
kernel = sst.gaussian_kde(toclust_f5.loc[~toclust_f5.sw.isnull(), 'sw'], 0.1)
x = np.linspace(0, 1.5, 100)
hist = kernel(x)

# find valley in bimodal distribution
min_idx = snl.argrelmin(hist)[0]
hist_threshold = x[min_idx[0]]
margin = 0.05  # plus or minus in ms

# Classifies base on valley plus an unclasified zone of 0.1ms
named_labels = np.empty(len(toclust_f5['sw']), dtype=object)
named_labels[toclust_f5['sw'] < (hist_threshold - margin)] = 'narrow'
named_labels[
    np.logical_and(
        (hist_threshold - margin) <= toclust_f5['sw'],
        (toclust_f5['sw'] < (hist_threshold + margin))
    )
] = 'unclass'
named_labels[(hist_threshold + margin) <= toclust_f5['sw']] = 'broad'

toclust_f5['sw_kde'] = named_labels

# copy over spike width classification
# and overwrites when neurons are optotagged
toclust_f5['triple'] = toclust_f5['sw_kde']
toclust_f5.loc[toclust_f5.phototag == 'a', 'triple'] = 'activated'
toclust_f5['triple'] = toclust_f5.triple.astype('category')

# ToDo I have the gut feeling this is an unecesarilly complicated DF and I
# might do with the simpler minimal dataframe

# Cluster Mass significant contextual effects
summary_DF_file = pl.Path(
    config['paths']['analysis_cache']) / f'220818_abs_deltaFR_DF'

# load integral across time chunks and whole lenght
metrics = ['integral', 'last_bin', 'integral_nosig']
DF_f5 = jl.load(summary_DF_file).query(
    f"source == 'real' and metric in {metrics} "
    f"and cluster_threshold == 0.05 and mult_comp_corr == 'bf_cp' "
    f"and analysis == 'SC' "
)

DF_f5.drop(columns=['source', 'cluster_threshold', 'mult_comp_corr',
                    'analysis', ], inplace=True)
DF_f5['metric'] = DF_f5.metric.cat.remove_unused_categories()
DF_f5.reset_index(drop=True, inplace=True)

# use the integrale value to find instances that are significant, and a new significance column
signif_filter = DF_f5.query("metric == 'integral' and value > 0").loc[:,
                ['id', 'context_pair', 'probe']].copy()
signif_filter['significant'] = True

DF_f5 = pd.merge(DF_f5, signif_filter, on=['id', 'context_pair', 'probe'],
                 how='left', validate='m:1')
DF_f5.loc[DF_f5.significant.isnull(), 'significant'] = False
DF_f5['significant'] = DF_f5.significant.astype(bool)

for col in ['id', 'context_pair', 'probe', 'site', 'region', 'metric',
            'stim_count']:
    DF_f5[col] = DF_f5[col].astype('category')

DF_f5['value'] = pd.to_numeric(DF_f5.value, downcast='float')
normalizer = DF_f5.groupby(by=['metric'], observed=True).agg(
    grand_mean=('value', np.mean)).reset_index()

DF_f5 = pd.merge(DF_f5, normalizer, on=['metric'], validate="m:1")
DF_f5['norm_val'] = DF_f5['value'] / DF_f5['grand_mean']
DF_f5.drop(columns=['grand_mean'], inplace=True)

# add celltype labels
toregress_f5 = pd.merge(DF_f5,
                        toclust_f5,
                        on='id', validate="m:1").reset_index(drop=True)

categories = ['activated', 'narrow', 'broad']
toplot_f5 = toregress_f5.query(f"triple in {categories} "
                               "and metric in ['integral', 'last_bin']"
                               "and significant"
                               ).copy()
