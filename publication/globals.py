import pathlib as pl
from configparser import ConfigParser

import joblib as jl
import numpy as np
import pandas as pd

from src.utils.dataframes import add_classified_contexts, norm_by_mean, \
    simplify_classified_contexts, add_ctx_type_voc
from src.root_path import config_path


# ToDo Start simplifying this mess of dataframes
###################### figure 2 ###################################

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 20, 'zscore': True, 'stim_type': 'permutations', }
alpha = 0.05
montecarlos = 11000
summary_DF_file = pl.Path(
    config['paths']['analysis_cache']) / '220520_minimal_DF'
metrics = ['integral', 'last_bin']
DF = jl.load(summary_DF_file).query(
    f"source == 'real' and metric in {metrics} and "
    f"cluster_threshold == 0.05 and mult_comp_corr == 'bf_cp' and "
    f"analysis == 'SC' and "
    f"diff_metric == 'delta_FR' and "
    f"value > 0")

DF.loc[DF.metric == 'integral', 'value'] = DF.loc[
                                               DF.metric == 'integral', 'value'] / 1000  # ms to s for better display

DF.drop(columns=['source', 'cluster_threshold', 'mult_comp_corr', 'analysis',
                 'stim_count', ], inplace=True)
DF.reset_index(drop=True, inplace=True)
DF = add_classified_contexts(DF)

for col in ['id', 'context_pair', 'probe', 'site', 'region', 'metric',
            'trans_pair']:
    DF[col] = DF[col].astype('category')

DF['value'] = pd.to_numeric(DF.value, downcast='float')

# pivots for the scatter
pivoted = DF.pivot_table(
    index=['region', 'site', 'context_pair', 'probe', 'trans_pair', 'id'],
    columns=['metric'], values='value', aggfunc='first',
    observed=True).reset_index()

# adds a small amount of jitter to the last bin value to help visualization
binsize = 1 / meta['raster_fs']
jitter = (np.random.random(pivoted.shape[0]) * binsize * 0.8 - (
        binsize * 0.8 / 2)) * 1000  # in ms
pivoted['last_bin_jittered'] = pivoted['last_bin'] + jitter


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

toregress = norm_by_mean(ferret_df)
toregress = simplify_classified_contexts(toregress)

toplot = list()  # concatenate data classified on single values
for cat in ['diff', 'same', 'silence']:
    subset = toregress.query(f"{cat} == 1").copy()
    subset['transition'] = cat
    toplot.append(subset)

toplot = pd.concat(toplot)

ferret_df = add_ctx_type_voc(ferret_df)

###################### figure 2 ###################################

