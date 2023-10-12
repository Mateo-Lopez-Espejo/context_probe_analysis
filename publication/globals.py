import pathlib as pl
from configparser import ConfigParser

import joblib as jl
import numpy as np
import pandas as pd
import scipy.stats as sst
import scipy.signal as snl

from nems.db import batch_comp

from src.utils.dataframes import add_classified_contexts, norm_by_mean, \
    simplify_classified_contexts, add_ctx_type_voc
from src.root_path import config_path
from src.utils.subsets import cellid_A1_fit_set, cellid_PEG_fit_set
from src.models.modelnames import modelnames as all_modelnames

from src.pupil.dataframes import (
    _calculate_pupil_first_order_coefficient,
    _calculate_delta_firing_rates,
    _calculate_pupil_second_order_coefficient,
    _filter_by_instance_significance
)

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))



# Baseline minimal parameters
RASTER_META = {'reliability': 0.1,  # r value
               'smoothing_window': 0,  # ms
               'raster_fs': 20,
               'zscore': True,
               'stim_type': 'permutations'}



MINIMAL_DF_FILE = pl.Path(
    config['paths']['analysis_cache']) / '220520_minimal_DF'


# attempt at filtering out a common minimal dataframe. It turns out the
# existing dataframe is already pretty minimal and this is a superfluous step
MINIMAL_DF = jl.load(MINIMAL_DF_FILE).query(
    f"source == 'real' "
    f"and cluster_threshold == 0.05 "
    f"and diff_metric == 'delta_FR' "
).drop(columns=['source', 'cluster_threshold', 'diff_metric'])

# integral values in seconds rather than ms for readability
MINIMAL_DF.loc[
    MINIMAL_DF.metric == 'integral', 'value'
] = MINIMAL_DF.loc[
        MINIMAL_DF.metric == 'integral', 'value'
    ] / 1000


# ToDo Start simplifying this mess of dataframes
###################### figure 2 ###################################

DF_f2 = MINIMAL_DF.query(
    f"metric in ['integral', 'last_bin'] "
    f"and mult_comp_corr == 'bf_cp' "
    f"and analysis == 'SC' "
    f"and value > 0 "
).drop(
    columns=['mult_comp_corr', 'analysis', 'stim_count']
).reset_index(
    drop=True
)

DF_f2 = add_classified_contexts(DF_f2)


# pivots for the scatter
pivoted_f2 = DF_f2.pivot_table(
    index=['region', 'site', 'context_pair', 'probe', 'trans_pair', 'id'],
    columns=['metric'], values='value', aggfunc='first',
    observed=True).reset_index()

# adds a small amount of jitter to the last bin value to help visualization
binsize = 1 / RASTER_META['raster_fs']
jitter = (np.random.random(pivoted_f2.shape[0]) * binsize * 0.8 - (
        binsize * 0.8 / 2)) * 1000  # in ms
pivoted_f2['last_bin_jittered'] = pivoted_f2['last_bin'] + jitter

# load and format ferret vocalization dataframe ToDo clean DFs
relevant_cols = ['id', 'region', 'named_ctx_0', 'named_ctx_1', 'named_probe',
                 'context_pair', 'probe', 'voc_ctx', 'voc_prb', 'metric',
                 'value']
# todo clean up the code that generates this,
# variable 'merged' in 221205_ferret_vocalization_effects.ipynb
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


DF_f3 = jl.load(MINIMAL_DF_FILE)

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

###################### figure 5 ###################################

# todo cleanup and organize these classifications to hold just the necessary

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


###################### figure 6 ###################################


model_df_file = pl.Path(config['paths']['analysis_cache']) / f'220412_resp_pred_metrics_by_chunks'
working_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220531_fig6_wdf'

recache_wdf = False

if working_DF_file.exists() and not recache_wdf:
    MODEL_CTX_QUANT_DF = jl.load(working_DF_file)
    print("found and loaded working data frame from cache")
else:
    print('creating working dataframe ...')

    filter = jl.load(MINIMAL_DF_FILE).query(
        f"mult_comp_corr == 'bf_cp' and source == 'real' and cluster_threshold == 0.05 and "
        f"metric == 'integral' and analysis == 'SC'"
        f"and value > 0 "
        ).loc[:,
             ['id', 'context_pair', 'probe']]  # these are the minimum columns to define an individual instance

    MODEL_CTX_QUANT_DF = jl.load(model_df_file)

    MODEL_CTX_QUANT_DF = pd.merge(
        filter, MODEL_CTX_QUANT_DF, on=['id', 'context_pair', 'probe'], validate='1:m'
    ).query("metric in ['integral', 'mass_center']").rename(columns={'resp': 'response'})

    # del (DF, filter)
    jl.dump(MODEL_CTX_QUANT_DF, working_DF_file)
    print('done')


DISPLAY_NAME_MAP = {'matchl_STRF': 'STRF',
                    'matchl_self': 'Self',
                    'matchl_pop': 'Pop',
                    'matchl_full': 'Full'}


MODEL_STATISTIC = 'r_test'
cellids = cellid_A1_fit_set.union(cellid_PEG_fit_set)
MODEL_NICKNAMES = ['matchl_STRF', 'matchl_self', 'matchl_pop', 'matchl_full']
modelnames = [mod for key, mod in all_modelnames.items() if
              key in MODEL_NICKNAMES]

DF = batch_comp(batch=[326, 327], modelnames=modelnames, cellids=cellids,
                stat=MODEL_STATISTIC)
DFfloor = batch_comp(batch=[326, 327], modelnames=modelnames,
                     cellids=cellids, stat='r_floor')

# nan out prediciton values smaller than r-floor,
# arr is a view of the DF data, the replacement is in place
arr = DF.values
arr[DF.values <= DFfloor.values] = np.nan

MODEL_DISPLAY_NAMES = [DISPLAY_NAME_MAP[nknm] for nknm in MODEL_NICKNAMES]
DF.columns = MODEL_DISPLAY_NAMES
DFfloor.columns = MODEL_DISPLAY_NAMES
DF.reset_index(inplace=True)
DFfloor.reset_index(inplace=True)
MODEL_PERFORMANCE_WIDE_DF = DF.dropna().rename(columns={'cellid': 'id', 'siteid': 'site'}),

###################### Suplementary figure 3 ##################################
# todo rename variables to coincide with final figure numeration

sup_fig3_wdf_file = pl.Path(
    config['paths']['analysis_cache']
) / f'230829_sup_fig3_wdf'

recache_pupil_wdf = False

if sup_fig3_wdf_file.exists() and not recache_pupil_wdf:
    SUP_FIG3_WDF = jl.load(sup_fig3_wdf_file)
    print("found and loaded pupil effects working data frame from cache")
else:
    print('creating working dataframe for pupil effects ...')

    # dataframe containing firing rates for isntances and time intervals
    fr_DF_file = pl.Path(
        config['paths']['analysis_cache']) / f'220808_pupil_fr_by_instance'

    # dataframe containing cluster mass context effects values by integral.
    filter_DF_file = pl.Path(
        config['paths']['analysis_cache']) / f'220719_chunked_amplitude_DF'

    pivoted_fo = _calculate_pupil_first_order_coefficient(
        jl.load(fr_DF_file)
    )

    second_ord_DF = _calculate_delta_firing_rates(
        jl.load(fr_DF_file)
    )

    pivoted_so = _calculate_pupil_second_order_coefficient(second_ord_DF)

    SUP_FIG3_WDF = _filter_by_instance_significance(
        pd.merge(pivoted_fo, pivoted_so, on=['id', 'site', 'probe'],
                 suffixes=('_fo', '_so'), validate='1:m'),
        jl.load(filter_DF_file)
    )


    jl.dump(SUP_FIG3_WDF, sup_fig3_wdf_file)
    print('done')
