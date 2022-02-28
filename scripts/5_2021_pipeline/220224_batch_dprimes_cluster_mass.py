from src.metrics.significance import _significance
from src.metrics.consolidated_dprimes import single_cell_dprimes_cluster_mass
from src.data.load import get_site_ids
from src.metrics.consolidated_metrics import metrics_to_DF
from src.data.region_map import region_map
from src.root_path import  config_path

import itertools as itt
import numpy as np
import numpy.ma as ma
import pandas as pd
from configparser import ConfigParser
import pathlib as pl
import joblib as jl

"""
this is the batch running script after a lot of refactoring, still should be runned by the cluster... work in progress
it is meant to run the streamlined analysis after choosing some of the relevant meta parameters
    4 and 10 sound
    permutations
    3 and 4 contiguous mutiple comparisons
    absolute significant integral
    absolute significatn center of mass
"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

meta = {'alpha': 0.05,
        'montecarlo': 1000,
        'raster_fs': 30,
        'reliability': 0.1,
        'smoothing_window': 0,
        'stim_type': 'permutations',
        'zscore': True}

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220224_ctx_mod_metric_DF_cluster_mass'
summary_DF_file.parent.mkdir(parents=True, exist_ok=True)

analysis_functions = {'SC': single_cell_dprimes_cluster_mass}

cluster_thresholds = [0.5,1.0,2.0]

expt = {'contexts': 'all',
        'probes': 'all'}

multiple_corrections = {'bf_cp': ([1,2], 0),
                        'none': (None, 0)}

metrics = ['significant_abs_mass_center', 'significant_abs_sum']

sites = set(get_site_ids(316).keys())
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a', 'TNC010a'} # empirically decided
no_perm = {'ley058d'} # sites without permutations
sites = sites.difference(badsites).difference(no_perm)
print(f'all sites: \n{sites}\n')
# sites = ('AMT021b',) # test site


to_concat = list()

if summary_DF_file.exists():
    DF = jl.load(summary_DF_file)
    ready_sites = set(DF.siteid.unique())
    sites = sites.difference(ready_sites)
    print('appening new sites to existing DF', sites)
    to_concat.append(DF)

for site, (fname, func), clust_thresh in itt.product(
        sites, analysis_functions.items(), cluster_thresholds):
    print(f'\n########################'
          f'\noutmost loop, working on site {site}, {fname}\n'
          f'########################\n')

    dprime, clust_quant_pval, goodcells, shuffled_eg = func(site, contexts='all', probes='all',
                                                            cluster_threshold=clust_thresh, meta=meta)

    # for analysis with dimensionality reduction, changes the cellname to nan for proper dimension labeling.
    if 'SC' in fname:
        chan_name = goodcells
    else:
        chan_name = [np.nan]

    # literal contexts and probe for dimlabdict
    contexts = list(range(0,dprime.shape[2]+1))
    probes = list(range(1,dprime.shape[2]+1))


    # creates label dictionalry
    dim_labl_dict = {'cellid': chan_name,
                    'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(contexts, 2)],
                    'probe': probes,
                    'time': np.linspace(0, dprime.shape[-1] / meta['raster_fs'], dprime.shape[-1],
                                        endpoint=False) * 1000}

    # calculates different significance corrections
    # calculate significant time bins, both raw and corrected for multiple comparisons
    for corr_name, (corr, cons) in multiple_corrections.items():
        print(f'    comp_corr: {corr_name}')

        # iterates over real data and shuffled example

        for source in ['real', 'shuffled_eg']:
            if source == 'real':
                dp = dprime
                pvals = clust_quant_pval
            elif source == 'shuffled_eg':
                dp = shuffled_eg['dprime']
                pvals =  {'pvalue':shuffled_eg['pvalue']}

            significance = _significance(dp, pvals, corr, cons, alpha=meta['alpha'])

            masked_dprime = ma.array(dp, mask=significance == 0)

            df = metrics_to_DF(masked_dprime, dim_labl_dict, metrics=metrics)
            df['mult_comp_corr'] = corr_name
            df['analysis'] = fname
            df['siteid'] = site
            df['region'] = region_map[site]
            df['source'] = source
            df['cluster_threshold'] = clust_thresh

            to_concat.append(df)

DF = pd.concat(to_concat, ignore_index=True, axis=0)

print(f'duplicated columns: {np.sum(DF.duplicated().values)}')
DF.drop_duplicates(inplace=True)
print(DF.head(10))

jl.dump(DF, summary_DF_file)
