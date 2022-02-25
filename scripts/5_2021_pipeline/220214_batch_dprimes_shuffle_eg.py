from src.metrics.significance import _significance
from src.metrics.consolidated_dprimes import single_cell_dprimes, probewise_dPCA_dprimes, full_dPCA_dprimes
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
similar to prior examples, here it parses an extra random shuffle example for
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

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220214_ctx_mod_metric_DF'
summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220214_ctx_mod_metric_DF_MCC'
summary_DF_file.parent.mkdir(parents=True, exist_ok=True)

analysis_functions = {'SC': single_cell_dprimes}

expt = {'contexts': 'all',
        'probes': 'all'}

multiple_corrections = {'bf_cpt': ([1,2,3], 0),
                        'bf_ncpt': ([0,1,2,3], 0),
                        'bf_t': ([3], 0),
                        'consecutive_3': ([3], 3)}

metrics = ['significant_abs_mass_center', 'significant_abs_sum']

sites = set(get_site_ids(316).keys())
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a' } # empirically decided
no_perm = {'ley058d'}
sites = sites.difference(badsites).difference(no_perm)
print(f'all sites: \n{sites}\n')
# sites = set(('TNC010a',))

if summary_DF_file.exists():
    DF = jl.load(summary_DF_file)
    ready_sites = set(DF.siteid.unique())
    sites = sites.difference(ready_sites)
    print('appening new sites to existing DF', sites)
else:
    DF = pd.DataFrame()


for site, (fname, func) in itt.product(sites, analysis_functions.items()):
    print(f'\n########################'
          f'\noutmost loop, working on site {site}, {fname}\n'
          f'########################\n')

    dprime, pval_quantiles, goodcells, shuff_eg = func(site, contexts='all', probes='all', meta=meta)

    # pvalue = pval_quantiles['pvalue']
    # quantiles = {key:val for key,val in pval_quantiles.items() if key != 'pvalue'}

    # for analysis with dimensionality reduction, changes the cellname to nan for proper dimension labeling.
    if fname != 'SC':
        chan_name = [np.nan]
    else:
        chan_name = goodcells

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
            elif source == 'shuffled_eg':
                dp = shuff_eg['dprime']

            significance, confidence_interval = _significance(dp, pval_quantiles, corr, cons, alpha=meta['alpha'])

            masked_dprime = ma.array(dp, mask=significance == 0)

            df = metrics_to_DF(masked_dprime, dim_labl_dict, metrics=metrics)
            df['mult_comp_corr'] = corr_name
            df['analysis'] = fname
            df['siteid'] = site
            df['region'] = region_map[site]
            df['source'] = source

            DF = DF.append(df,ignore_index=True)

DF.drop_duplicates(inplace=True)


jl.dump(DF, summary_DF_file)
