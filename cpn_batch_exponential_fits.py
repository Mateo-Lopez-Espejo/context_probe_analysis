import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import joblib as jl
import numpy as np
import pandas as pd
from scipy.integrate import trapz

import cpn_dprime as cDP
import fits as fts
from cpp_cache import set_name

"""
takes all the dprimes and pvalues, fits exponential decays to both the dprimes and the profiles of dprime
significance (derived from pvalues). 
This is done for all combinations of probe, and context transtion pairs.
This is done for single cells, dPCA and LDA.
"""

config = ConfigParser()
if pl.Path('../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../context_probe_analysis/config/settings.ini'))
elif pl.Path('../../../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../../../context_probe_analysis/config/settings.ini'))
else:
    raise FileNotFoundError('config file could not be find')

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None}

region_map = dict(
    zip(['AMT028b', 'AMT029a', 'AMT030a', 'AMT031a', 'AMT032a', 'DRX008b', 'DRX021a', 'ley070a', 'ley072b'],
        ['PEG', 'PEG', 'PEG', 'PEG', 'PEG', 'A1', 'A1', 'A1', 'A1']))

recache = False

# loads the raw calculated dprimes and montecarlos
batch_dprime_file = pl.Path(config['paths']['analysis_cache']) / 'batch_dprimes' / set_name(meta)
batch_dprimes = jl.load(batch_dprime_file)

# defines significant values based on loaded pvalue and defined threshold
threshold = 0.01
for analysis_name, mid_dict in batch_dprimes.items():
    mid_dict['shuffled_significance'] = {key: (val <= threshold) for key, val in mid_dict['shuffled_pvalue'].items()}
    if analysis_name != 'SC':
        mid_dict['simulated_significance'] = {key: (val <= threshold) for key, val in
                                              mid_dict['simulated_pvalue'].items()}

# set up the time bin labels in milliseconds, this is critical for plotting and calculating the tau
nbin = np.max([value.shape[-1] for value in batch_dprimes['SC']['dprime'].values()])
times = np.linspace(0, nbin / meta['raster_fs'], nbin, endpoint=False) * 1000

sites = set(batch_dprimes['dPCA']['dprime'].keys())
all_probes = [2, 3, 5, 6]
all_trans = [f'{t0}_{t1}' for t0, t1 in itt.combinations(meta['transitions'], 2)]

# creates and caches, or loads the DF

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / 'DF_summary' / set_name(meta)
if summary_DF_file.exists() is False or recache is True:

    print('creating summary DataFrame')
    df = list()
    for analysis, source_dict in batch_dprimes.items():
        print(f'\nanalysis {analysis}')

        for source, id_dict in source_dict.items():
            if source == 'dprime':
                pass
            elif source == 'shuffled_significance':
                source = 'significance'
            else:
                continue
            print(f'\nsource {source}\n')

            for id, array in id_dict.items():
                print(f'{id}')

                # flips each time series in the dprime array, so the max absolute value is positive
                if source == 'dprime':
                    array, _ = cDP.flip_dprimes(array, None, flip='max')

                t = times[:array.shape[-1]]
                # defines if working with cell or site id, infers site form cells
                if len(id) > 7:
                    cell = id
                    site = id[:7]
                elif len(id) == 7:
                    cell = np.nan
                    site = id
                else:
                    raise ValueError('id dont match to site of cell format')

                # preorganizes means across probes or context pairs prior to further different metric calculations
                mean_dicts = list()
                # mean of transition pairs for each probe
                for pp, probe in enumerate(all_probes):
                    mean = np.mean(array[pp, :, :], axis=0)
                    mean_dicts.append({'probe': f'probe_{probe}', 'transition_pair': 'mean', 'mean': mean})

                # mean of probes for each transition pair
                for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
                    mean = np.mean(array[:, tt, :], axis=0)
                    mean_dicts.append({'probe': 'mean', 'transition_pair': f'{trans[0]}_{trans[1]}', 'mean': mean})

                # full mean across probes and transition pairs
                mean = np.mean(array[:, :, :], axis=(0, 1))
                mean_dicts.append({'probe': 'mean', 'transition_pair': 'mean', 'mean': mean})

                # calculates different metrics/parameters for each different mean
                for mean_dict in mean_dicts:

                    # parameters  is a list of dictionaries containing the parameter name, the value and goodness of fit
                    parameters = list()

                    # exponential decay fit
                    popt, r2 = fts.exp_decay(t, mean_dict['mean'], skip_error=True)
                    parameters.append({'parameter': 'r0', 'value': popt[0], 'goodness': r2})
                    parameters.append({'parameter': 'tau', 'value': -1 / popt[1], 'goodness': r2})

                    # max value
                    parameters.append({'parameter': 'max', 'value': np.max(mean_dict['mean']), 'goodness': np.nan})

                    # full dprime integral
                    if source == 'dprime':
                        parameters.append(
                            {'parameter': 'integral', 'value': trapz(mean_dict['mean'], t), 'goodness': np.nan})

                    # significant dprime integral
                    if source == 'dprime':
                        signif_array = batch_dprimes[analysis]['shuffled_significance'][id]

                        # collapses significance across the same categories used to get the dprime mean
                        if mean_dict['probe'] == 'mean' and mean_dict['transition_pair'] != 'mean':
                            signinf_mean = np.mean(signif_array, axis=0)[all_trans.index(mean_dict['transition_pair']),
                                           :]
                        elif mean_dict['probe'] != 'mean' and mean_dict['transition_pair'] == 'mean':
                            signinf_mean = np.mean(signif_array, axis=0)[all_probes.index(int(mean_dict['probe'][-1])),
                                           :]
                        elif mean_dict['probe'] == 'mean' and mean_dict['transition_pair'] == 'mean':
                            signinf_mean = np.mean(signif_array, axis=(0, 1))
                        else:
                            raise ValueError('unrecongnized mean pattern')

                        signif_mask = signinf_mean > 0
                        parameters.append({'parameter': 'significant_integral',
                                           'value': trapz(mean[signif_mask], t[signif_mask]), 'goodness': np.nan})

                    for parameter in parameters:
                        d = {'siteid': site,
                             'cellid': cell,
                             'analysis': analysis,  # singel_cell, dPCA
                             'probe': mean_dict['probe'],  # probe_n, mean
                             'transition_pair': mean_dict['transition_pair'],  # t0_t1, mean
                             'parameter': parameter['parameter'],
                             'goodness': parameter['goodness'],
                             'source': source,
                             'value': parameter['value']}
                        df.append(d)

    DF = pd.DataFrame(df)
    # add brain region
    DF['region'] = [region_map[site] for site in DF.siteid]
    del df

    if summary_DF_file.parent.exists() is False:
        summary_DF_file.parent.mkdir()
    _ = jl.dump(DF, summary_DF_file)

else:
    print('loading cached DF')
    DF = jl.load(summary_DF_file)
