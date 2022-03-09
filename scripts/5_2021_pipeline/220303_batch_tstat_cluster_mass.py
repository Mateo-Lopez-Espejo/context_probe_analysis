import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import numpy as np
import pandas as pd
import joblib as jl

from src.data.load import get_site_ids
from src.data.region_map import region_map
from src.metrics.consolidated_tstat import single_cell_tstat_cluster_mass
from src.metrics.significance import _significance
from src.metrics.time_series_summary import metrics_to_DF
from src.root_path import config_path
from src.utils.dataframes import add_classified_contexts, ndim_array_to_long_DF

"""

"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'stim_type': 'permutations'}

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220303_ctx_mod_metric_DF_tstat_cluster_mass'
summary_DF_file.parent.mkdir(parents=True, exist_ok=True)

analysis_functions = {'SC': single_cell_tstat_cluster_mass}

alpha = 0.05
cluster_thresholds = [0.05, 0.01]
multiple_corrections = {'bf_cp': ([1, 2], 0),
                        'bf_ncp': ([0, 1, 2], 0),
                        'none': (None, 0)}

metrics = ['mass_center', 'integral', 'last_bin']

sites = set(get_site_ids(316).keys())
badsites = {'AMT031a', 'DRX008b', 'DRX021a', 'DRX023a', 'ley074a', 'TNC010a'}  # empirically decided
no_perm = {'ley058d'}  # sites without permutations
sites = sites.difference(badsites).difference(no_perm)
# sites = ('AMT029a', )
print(f'all sites: \n{sites}\n')


recacheDF = True

if summary_DF_file.exists() and not recacheDF:
    DF = jl.load(summary_DF_file)
    ready_sites = set(DF.siteid.unique())
    sites = sites.difference(ready_sites)
    print('appening new sites to existing DF', sites)
    to_concat = [DF,]
else:
    to_concat = list()

for site, (fname, func), clust_thresh in itt.product(
        sites, analysis_functions.items(), cluster_thresholds):
    print(f'noutmost loop, working on site {site}, {fname}')

    if func.check_call_in_cache(site, contexts='all', probes='all',
                                cluster_threshold=float(clust_thresh), meta=meta):
        tstat, clust_quant_pval, goodcells, shuffled_eg = func(site, contexts='all', probes='all',
                                                               cluster_threshold=float(clust_thresh), meta=meta)
    else:
        print(f'{site}, {func}, {clust_thresh} not yet in cache, skipping')
        continue

    # for analysis with dimensionality reduction, changes the cellname to nan for proper dimension labeling.
    if 'SC' in fname:
        chan_name = goodcells
    else:
        chan_name = site

    # literal contexts and probe for dimlabdict
    contexts = list(range(0, tstat.shape[2] + 1))
    probes = list(range(1, tstat.shape[2] + 1))

    # creates label dictionalry
    dim_labl_dict = {'id': chan_name,
                     'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(contexts, 2)],
                     'probe': probes,
                     'time': np.linspace(0, tstat.shape[-1] / meta['raster_fs'], tstat.shape[-1],
                                         endpoint=False) * 1000}


    # iterates over real data and shuffled example
    for source in ['real', 'shuffled_eg']:
        # consider different multiple comparisons corrections for the significance dependent metrics
        for corr_name, (corr, cons) in multiple_corrections.items():
            if source == 'real':
                ts = tstat
                pvals = clust_quant_pval['pvalue']
            elif source == 'shuffled_eg':
                ts = shuffled_eg['dprime']
                pvals = shuffled_eg['pvalue']

            significance = _significance(pvals, corr, cons, alpha=alpha)

            masked_dprime = np.ma.array(ts, mask=significance == 0)

            df = metrics_to_DF(masked_dprime, dim_labl_dict, metrics=metrics)
            df['mult_comp_corr'] = corr_name
            df['analysis'] = fname
            df['site'] = site
            df['region'] = region_map[site]
            df['source'] = source
            df['cluster_threshold'] = clust_thresh
            df['stim_count'] = len(probes)

            to_concat.append(df)

        # keeps raw p values
        pval_lbl_dict = dim_labl_dict.copy()
        pval_lbl_dict.pop('time')
        min_pval = np.min(pvals, axis=-1)
        df = ndim_array_to_long_DF(min_pval, pval_lbl_dict)
        df['metric'] = 'pvalue'
        df['analysis'] = fname
        df['site'] = site
        df['region'] = region_map[site]
        df['source'] = source
        df['cluster_threshold'] = clust_thresh
        df['stim_count'] = len(probes)

        to_concat.append(df)

DF = pd.concat(to_concat, ignore_index=True, axis=0)

# extra formatting
print(f'adding context clasification')
DF = add_classified_contexts(DF)

dups = np.sum(DF.duplicated().values)
if dups > 0:
    print(f'{dups} duplicated rows, what is wrong?, droping duplicates')
    DF.drop_duplicates(inplace=True)

print(DF.head(10))
jl.dump(DF, summary_DF_file)
