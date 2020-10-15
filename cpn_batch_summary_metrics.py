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
elif pl.Path('/home/mateo/code/context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('/home/mateo/code/context_probe_analysis/config/settings.ini'))
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

recache = True

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
                # individual values i.e. no mean
                for (pp, probe), (tt, trans) in itt.product(enumerate(all_probes),
                                                            enumerate(itt.combinations(meta['transitions'], 2))):
                    mean = array[pp, tt, :]
                    mean_dicts.append({'probe': f'probe_{probe}', 'transition_pair': f'{trans[0]}_{trans[1]}', 'value': mean})

                # mean of transition pairs for each probe
                for pp, probe in enumerate(all_probes):
                    mean = np.mean(array[pp, :, :], axis=0)
                    mean_dicts.append({'probe': f'probe_{probe}', 'transition_pair': 'mean', 'value': mean})

                # mean of probes for each transition pair
                for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
                    mean = np.mean(array[:, tt, :], axis=0)
                    mean_dicts.append({'probe': 'mean', 'transition_pair': f'{trans[0]}_{trans[1]}', 'value': mean})

                # full mean across probes and transition pairs
                mean = np.mean(array[:, :, :], axis=(0, 1))
                mean_dicts.append({'probe': 'mean', 'transition_pair': 'mean', 'value': mean})

                # calculates different metrics/parameters for each different mean
                for mean_dict in mean_dicts:

                    mean = mean_dict.pop('value') # pops to not include array in dataframe later on

                    # parameters  is a list of dictionaries containing the parameter name, the value and goodness of fit
                    parameters = list()

                    # exponential decay fit
                    popt, pcov, r2 = fts.exp_decay(t, mean, skip_error=True)
                    perr = np.sqrt(np.diag(pcov)) # parameter standard deviation
                    parameters.append({'parameter': 'r0', 'std': perr[0], 'goodness': r2, 'value': popt[0]})
                    parameters.append({'parameter': 'tau', 'std': 1/perr[1], 'goodness': r2, 'value': -1/popt[1]})

                    # max value
                    parameters.append({'parameter': 'max', 'value': np.max(mean).astype(float)})

                    if source == 'dprime':
                        # full dprime integral
                        parameters.append(
                            {'parameter': 'integral', 'value': trapz(mean, t)})
                        # full dprime sum
                        parameters.append(
                            {'parameter': 'sum', 'value': np.sum(mean) * np.mean(np.diff(t))})
                        # full dprime absolute sum
                        parameters.append(
                            {'parameter': 'abs_sum', 'value': np.sum(np.abs(mean)) * np.mean(np.diff(t))})

                        #full dprime center of mass
                        parameters.append(
                            {'parameter': 'mass_center', 'value': np.sum(mean * t) / np.sum(mean)})

                        #full dprime absolute center of mass
                        parameters.append(
                            {'parameter': 'abs_mass_center', 'value': np.sum(np.abs(mean) * t) / np.sum(np.abs(mean))})


                        # gets the significant bins from the full data array and take the mean across categories as needed
                        signif_array = batch_dprimes[analysis]['shuffled_significance'][id]
                        if mean_dict['probe'] == 'mean' and mean_dict['transition_pair'] != 'mean':
                            signinf_mean = np.mean(signif_array, axis=0)[all_trans.index(mean_dict['transition_pair']),:]
                        elif mean_dict['probe'] != 'mean' and mean_dict['transition_pair'] == 'mean':
                            signinf_mean = np.mean(signif_array, axis=1)[all_probes.index(int(mean_dict['probe'][-1])),:]
                        elif mean_dict['probe'] == 'mean' and mean_dict['transition_pair'] == 'mean':
                            signinf_mean = np.mean(signif_array, axis=(0, 1))
                        elif mean_dict['probe'] != 'mean' and mean_dict['transition_pair'] != 'mean':
                            signinf_mean = signif_array[all_probes.index(int(mean_dict['probe'][-1])),
                                           all_trans.index(mean_dict['transition_pair']), :]
                        else:
                            raise ValueError(f"unknown {mean_dict['probe']} and or {mean_dict['transition_pair']}")

                        signif_mask = signinf_mean > 0

                        # significant dprime integral
                        parameters.append({'parameter': 'significant_integral',
                                           'value': trapz(mean[signif_mask], t[signif_mask])})
                        # significant dprime sum
                        parameters.append({'parameter': 'significant_sum',
                                           'value': np.sum(mean[signif_mask]) * np.mean(np.diff(t))})
                        # significant dprime absolute sum
                        parameters.append({'parameter': 'significant_abs_sum',
                                           'value': np.sum(np.abs(mean[signif_mask])) * np.mean(np.diff(t))})

                        # significant dprime center of mass
                        parameters.append({'parameter': 'significant_mass_center',
                                           'value': np.sum(mean[signif_mask] * t[signif_mask]) / np.sum(mean[signif_mask])})

                        # significant dprime absolute center of mass
                        parameters.append({'parameter': 'significant_abs_mass_center',
                                           'value': np.sum(np.abs(mean[signif_mask]) * t[signif_mask]) /
                                                    np.sum(np.abs(mean[signif_mask]))})

                    for parameter in parameters:
                        d = {'siteid': site,
                             'cellid': cell,
                             'analysis': analysis,  # singel_cell, dPCA
                             'source': source, # dprime, significance
                             **mean_dict,
                             **parameter}
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
