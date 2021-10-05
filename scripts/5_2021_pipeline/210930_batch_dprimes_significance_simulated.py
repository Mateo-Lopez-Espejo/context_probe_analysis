from src.metrics.significance import _significance
from src.metrics.consolidated_dprimes_simulation import single_cell_dprimes, probewise_dPCA_dprimes, full_dPCA_dprimes
from src.metrics.dprime import flip_dprimes
from src.data.load import set_name
from src.metrics.consolidated_metrics import metrics_to_DF
from src.data.region_map import region_map
from src.root_path import  config_path

import itertools as itt
import numpy as np
import numpy.ma as ma
import pandas as pd
from configparser import ConfigParser
import pathlib as pl
from joblib import dump, load, Parallel, delayed

"""
this version proceses the 10 sound permutation dataset.
Only works for permutations

aditionaly it takes simulated contextual modulation
"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

rec_recache = False
dprime_recache = False

signif_tails = 'both'
alpha=0.05

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None,
        'stim_type': 'permutations'}

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'210930_consolidated_summary_DF_alpha_{alpha}_simulated' / set_name(meta)
variance_DF_file = pl.Path(config['paths']['analysis_cache']) / f'210930_variance_explained_DF_simulated' / set_name(meta)

analysis_functions = {'SC': single_cell_dprimes, #'LDA':probewise_LDA_dprimes,
                      'pdPCA': probewise_dPCA_dprimes, 'fdPCA': full_dPCA_dprimes}

expt = {'contexts': list(range(11)),
        'probes': list(range(1,11))}

multiple_corrections = {'consecutive_2': ([3], 2),
                        'consecutive_3': ([3], 3),
                        'consecutive_4': ([3], 4)}

metrics = ['significant_abs_mass_center', 'significant_abs_sum']

# sites = set(get_site_ids(316).keys())
sites = set(['TNC013a', 'TNC014a', 'TNC015a', 'TNC016a', 'TNC017a', 'TNC018a'])
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a' }  # empirically decided
sites = sites.difference(badsites)


# run main function on its own, paralelized and cached, to speed up later metric calculation and DF creation
_ = Parallel(n_jobs=7)(delayed(func)
                       (ss, **expt, meta=meta)
                        for func, ss in itt.product(analysis_functions.values(), sites))

if summary_DF_file.exists():
    DF = load(summary_DF_file)
    ready_sites = set(DF.siteid.unique())
    sites = sites.difference(ready_sites)
    print('appening new sites to existing DF', sites)
else:
    DF = pd.DataFrame()

bads = list()
for site, (fname, func) in itt.product(sites, analysis_functions.items()):

    dprime, shuff_dprime_quantiles, goodcells, var_capt = func(site, **expt, meta=meta)

    # for analysis with dimensionality reduction, changes the cellname to nan for proper dimension labeling.
    if fname != 'SC':
        chan_name = [np.nan]
    else:
        chan_name = goodcells

    # creates label dictionalry
    dim_labl_dict = {'cellid': chan_name,
                    'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(expt['contexts'], 2)],
                    'probe': expt['probes'],
                    'time': np.linspace(0, dprime.shape[-1] / meta['raster_fs'], dprime.shape[-1],
                                        endpoint=False) * 1000}

    # calculats different significaces/corrections
    # calculate significant time bins, both raw and corrected for multiple comparisons
    for corr_name, (corr, cons) in multiple_corrections.items():
        print(f'    comp_corr: {corr_name}')

        significance, confidence_interval = _significance(dprime, shuff_dprime_quantiles, corr, cons, alpha=alpha)
        fliped, _ = flip_dprimes(dprime, flip='sum')

        masked_dprime_means = ma.array(fliped, mask=significance == 0)

        df = metrics_to_DF(masked_dprime_means, dim_labl_dict, metrics=metrics)
        df['mult_comp_corr'] = corr_name
        df['analysis'] = fname
        df['siteid'] = site
        df['region'] = region_map[site]

        DF = DF.append(df,ignore_index=True)

print('failed sites: ', bads)

DF.drop_duplicates(inplace=True)

if summary_DF_file.parent.exists() is False:
    summary_DF_file.parent.mkdir()
dump(DF, summary_DF_file)


# calculates variance captuded by the full dPCA and organized in a DF
variance_DF = list()
for site in sites:

    _, _, _, var_capt = full_dPCA_dprimes(site, **expt, meta=meta)

    cum_var, dpc_var, marg_var, total_marginalized_var, comp_id = var_capt

    total_marginalized_var['siteid'] = site
    variance_DF.append(total_marginalized_var)

variance_DF = pd.DataFrame(variance_DF)


if variance_DF_file.parent.exists() is False:
    variance_DF_file.parent.mkdir()
dump(variance_DF, variance_DF_file)
