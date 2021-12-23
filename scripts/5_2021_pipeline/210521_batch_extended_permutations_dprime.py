from src.metrics.significance import _significance, _mask_with_significance
from src.metrics.consolidated_dprimes import single_cell_dprimes, probewise_LDA_dprimes, probewise_dPCA_dprimes, full_dPCA_dprimes
from src.metrics.dprime import flip_dprimes
from src.data.load import get_site_ids, set_name
from src.metrics.consolidated_metrics import metrics_to_DF, _append_means_to_array, _append_means_to_shuff_array
from src.data.region_map import region_map

import itertools as itt
import numpy as np
import numpy.ma as ma
import pandas as pd
from configparser import ConfigParser
import pathlib as pl
from joblib import dump, load

"""
This is a trimmed down version of previous pipelines in preparation for the extended permutation dataset, and having 
streamlined the most important metrics for my data. 
1. excludes mean significance policies
2. sticks to consecutive_3 Multiple comparisons corrections
3. excludes analysis for Triplets 
4. uses "all" keyword instead of explicit contexts and probes list
"""

config = ConfigParser()
config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))

rec_recache = False
dprime_recache = True

signif_tails = 'both'
alpha=0.05

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None}

#new
summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'210521_consolidated_summary_DF_alpha_{alpha}' / set_name(meta)
variance_DF_file = pl.Path(config['paths']['analysis_cache']) / f'210521_variance_explained_DF' / set_name(meta)

analysis_functions = {'SC': single_cell_dprimes,'LDA':probewise_LDA_dprimes,
                      'pdPCA': probewise_dPCA_dprimes, 'fdPCA': full_dPCA_dprimes}

permutations = {'contexts': 'all',
                'probes': 'all',
                'stim_type': 'permutations'}

triplets = {'contexts': ['silence', 'continuous', 'similar', 'sharp'],
            'probes':[2, 3, 5, 6],
            'stim_type': 'triplets'}

experiments = [permutations]

multiple_corrections = {'consecutive_3': ([3], 3)}

metrics = ['significant_abs_mass_center', 'significant_abs_sum']

sites = set(get_site_ids(316).keys())
# sites = {'ARM031a'}
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a' }  # empirically decided
sites = sites.difference(badsites)


if summary_DF_file.exists() and not dprime_recache:
    DF = load(summary_DF_file)
    ready_sites = set(DF.siteid.unique())
    sites = sites.difference(ready_sites)
    print('appening new sites to existing DF', sites)
else:
    DF = pd.DataFrame()

bads = list()
for site, expt, (fname, func) in itt.product(sites, experiments, analysis_functions.items()):

    # skips full_dPCA for the triplets experiment
    if expt['stim_type'] == 'triplets' and fname == 'fdPCA':
        continue

    print(site, expt['stim_type'], fname)

    # parses the stim_type from the experiment into the meta parameters
    expt = expt.copy()
    meta['stim_type'] = expt.pop('stim_type')

    # runs the dprime function
    # try:
    dprime, shuffled_dprime, goodcells, var_capt = func(site, **expt, meta=meta)
    # except:
    #     print('failed calculating dprimes')
    #     bads.append((site, meta['stim_type'], fname))

    # for analysis with dimensionality reduction, changes the cellname to nan for proper dimension labeling.
    if fname != 'SC':
        chan_name = [np.nan]
    else:
        chan_name = goodcells

    if expt['contexts'] == 'all':
        expt['contexts'] = list(range(0,dprime.shape[2]+1))

    if expt['probes'] == 'all':
        expt['probes'] = list(range(1,dprime.shape[2]+1))

    # creates label dictionayr with dimension corresponding to dprimes
    dim_lab_dict = {'cellid': chan_name,
                    'context_pair': [f'{c1}_{c2}' for c1, c2 in itt.combinations(expt['contexts'], 2)],
                    'probe': expt['probes'],
                    'time': np.linspace(0, dprime.shape[-1] / meta['raster_fs'], dprime.shape[-1],
                                        endpoint=False) * 1000}

    # multiple comparisosn correction for dprime significance acrosse timebins and/or probe, context_pair categories.
    for corr_name, (corr, cons) in multiple_corrections.items():

        print(f'    comp_corr: {corr_name}')
        significance, confidence_interval = _significance(dprime, shuffled_dprime, corr, cons, alpha=alpha)

        fliped, fliped_shuffled = flip_dprimes(dprime, shuffled_dprime, flip='sum')
        masked = ma.array(fliped, mask=significance == False)

        # calculate different metrics and organize into a dataframe
        df = metrics_to_DF(masked, dim_lab_dict, metrics=metrics)
        df['mult_comp_corr'] = corr_name
        df['stim_type'] = meta['stim_type']
        df['analysis'] = fname
        df['siteid'] = site
        df['region'] = region_map[site]

        DF = DF.append(df,ignore_index=True)

print('failed sites: ', bads)

DF.drop_duplicates(inplace=True)

if summary_DF_file.parent.exists() is False:
    summary_DF_file.parent.mkdir()
dump(DF, summary_DF_file)


# calculates variance captured by the full dPCA and organized in a DF
variance_DF = list()
for site in sites:
    expt = permutations.copy()
    meta['stim_type'] = expt.pop('stim_type')

    _, _, _, var_capt = full_dPCA_dprimes(site, **expt, meta=meta)

    cum_var, dpc_var, marg_var, total_marginalized_var, comp_id = var_capt

    total_marginalized_var['siteid'] = site
    variance_DF.append(total_marginalized_var)

variance_DF = pd.DataFrame(variance_DF)


if variance_DF_file.parent.exists() is False:
    variance_DF_file.parent.mkdir()
dump(variance_DF, variance_DF_file)
