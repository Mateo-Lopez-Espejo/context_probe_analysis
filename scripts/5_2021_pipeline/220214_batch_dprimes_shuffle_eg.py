from src.metrics.significance import _significance
from src.metrics.consolidated_dprimes import single_cell_dprimes, probewise_dPCA_dprimes, full_dPCA_dprimes
from src.metrics.dprime import flip_dprimes
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

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None,
        'stim_type': 'permutations',
        'alpha':0.05}

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220214_ctx_mod_metric_DF_{meta}'
summary_DF_file.parent.mkdir(parents=True, exist_ok=True)

# analysis_functions = {'SC': single_cell_dprimes, #'LDA':probewise_LDA_dprimes,
#                       'pdPCA': probewise_dPCA_dprimes, 'fdPCA': full_dPCA_dprimes}
analysis_functions = {'SC': single_cell_dprimes}

expt = {'contexts': 'all',
        'probes': 'all'}

# multiple_corrections = {'consecutive_3': ([3], 3),
#                         'consecutive_4': ([3], 4)}
multiple_corrections = {'consecutive_3': ([3], 3)}

metrics = ['significant_abs_mass_center', 'significant_abs_sum']

sites = set(get_site_ids(316).keys())
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a' } # empirically decided
no_perm = {'ley058d'}
sites = sites.difference(badsites).difference(no_perm)
print(f'all sites: \n{sites}\n')
sites = ('TNC014a', 'TNC008a', 'TNC010a')

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

    dprime, pval_quantiles, goodcells, shuff_eg = func(site, **expt, meta=meta)

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

    # calculats different significaces/corrections
    # calculate significant time bins, both raw and corrected for multiple comparisons
    for corr_name, (corr, cons) in multiple_corrections.items():
        print(f'    comp_corr: {corr_name}')

        significance, confidence_interval = _significance(dprime, pval_quantiles, corr, cons, alpha=meta['alpha'])
        fliped, _ = flip_dprimes(dprime, flip='sum')

        masked_dprime_means = ma.array(fliped, mask=significance == 0)

        df = metrics_to_DF(masked_dprime_means, dim_labl_dict, metrics=metrics)
        df['mult_comp_corr'] = corr_name
        df['analysis'] = fname
        df['siteid'] = site
        df['region'] = region_map[site]

        DF = DF.append(df,ignore_index=True)

DF.drop_duplicates(inplace=True)


jl.dump(DF, summary_DF_file)
