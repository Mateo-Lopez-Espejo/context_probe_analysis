from src.metrics.significance import _significance, _mask_with_significance
from src.metrics.consolidated_dprimes_v2 import single_cell_dprimes, probewise_LDA_dprimes, probewise_dPCA_dprimes, full_dPCA_dprimes
from src.metrics.dprime import flip_dprimes
from src.data.load import get_site_ids
from src.metrics.consolidated_metrics import metrics_to_DF, _append_means_to_array, _append_means_to_shuff_array
from src.data.load import set_name
from src.data.region_map import region_map

import itertools as itt
import numpy as np
import pandas as pd
from configparser import ConfigParser
import pathlib as pl
from joblib import dump

"""
I finaly realized that there is no need for weird mean significance policies. I can just take the mean of the flipped
shuffles, and use it to calculate significance (and CI) over the means. This is equivalent to shuffling and then taking 
the mean of the dprime n times. which is the sound approach to begin with.

1. Adds different corrections for multiple comparisons

2. Adds contiguous significance counting

3. Adds different ways of considering significance for the means of probes and context pairs, which I have
speculated in the past, might lead to over estimation of significance in late time bins.

4. Parses the variance captured by the full dPCA, so we can potentially see differences in the
marginalizations between regions.
"""

config = ConfigParser()
config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))

rec_recache = False
dprime_recache = False

signif_tails = 'both'
alpha=0.05

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None}

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'210302_consolidated_summary_DF_alpha_{alpha}' / set_name(meta)
variance_DF_file = pl.Path(config['paths']['analysis_cache']) / f'210302_variance_explained_DF' / set_name(meta)

analysis_functions = {'SC': single_cell_dprimes,'LDA':probewise_LDA_dprimes,
                      'pdPCA': probewise_dPCA_dprimes, 'fdPCA': full_dPCA_dprimes}


permutations = {'contexts': [0, 1, 2, 3, 4],
                'probes': [1, 2, 3, 4],
                'stim_type': 'permutations'}

triplets = {'contexts': ['silence', 'continuous', 'similar', 'sharp'],
            'probes':[2, 3, 5, 6],
            'stim_type': 'triplets'}

experiments = [permutations, triplets]

multiple_corrections = {'none': (None, None),
                        'time': ([3], None),
                        'full': ([1,2,3], None),
                        'consecutive_2': ([3], 2),
                        'consecutive_3': ([3], 3),
                        'consecutive_4': ([3], 4)}

mean_types = ['zeros', 'mean', 'shuffles']

metrics = ['significant_abs_mass_center', 'significant_abs_sum', 'significant_abs_mean']


sites = set(get_site_ids(316).keys())
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a' }  # empirically decided
sites = sites.difference(badsites)
# sites = ['CRD004a']

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
    try:
        dprime, shuffled_dprime, goodcells, var_capt = func(site, **expt, meta=meta)
    except:
        print('failed')
        bads.append((site, expt['stim_type'], fname))

    # for analysis with dimensionality reduction, changes the cellname to nan for proper dimension labeling.
    if fname != 'SC':
        chan_name = [np.nan]
    else:
        chan_name = goodcells

    # creates label dictionalry
    dim_lab_dict = {'cellid': chan_name,
                    'context_pair': [f'{c1}_{c2}' for c1, c2 in itt.combinations(expt['contexts'], 2)],
                    'probe': expt['probes'],
                    'time': np.linspace(0, dprime.shape[-1] / meta['raster_fs'], dprime.shape[-1],
                                        endpoint=False) * 1000}

    # calculats different significaces/corrections
    # calculate significant time bins, both raw and corrected for multiple comparisons
    for corr_name, (corr, cons) in multiple_corrections.items():
        print(f'    comp_corr: {corr_name}')

        significance, confidence_interval = _significance(dprime, shuffled_dprime, corr, cons, alpha=alpha)
        fliped, fliped_shuffled = flip_dprimes(dprime, shuffled_dprime, flip='sum')

        # only time correction for the means
        dprime_means, _  = _append_means_to_array(fliped, dim_lab_dict)
        shuff_means, _  = _append_means_to_shuff_array(fliped_shuffled, dim_lab_dict)
        shuf_signif, shuf_CI = _significance(dprime_means, shuff_means, [3], cons, alpha=alpha)

        for mean_type in mean_types:
            print(f'        mean_signif: {mean_type}')

            # masks dprime with different significances, uses different approaches to define significance of the mean.
            masked, masked_lab_dict = _mask_with_significance(fliped, significance, dim_lab_dict, mean_type=mean_type,
                                                              mean_signif_arr=shuf_signif)

            # calculate different metrics and organize into a dataframe
            df = metrics_to_DF(masked, masked_lab_dict, metrics=metrics)
            df['mult_comp_corr'] = corr_name
            df['mean_signif_type'] = mean_type
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


# calculates variance captuded by the full dPCA and organized in a DF
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
