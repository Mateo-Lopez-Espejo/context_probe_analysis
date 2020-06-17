import pandas as pd
from configparser import ConfigParser
import pathlib as pl
from cpp_cache import set_name
import numpy as np
import fits as fts
import itertools as itt
import joblib as jl

"""
takes all the fite dpriems and pvalues, fits exponential decays to both the dprimes and the profiles of dprime
significance. 
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

# loads the raw calculated dprimes and montecarlos
batch_dprime_file = pl.Path(config['paths']['analysis_cache']) / 'batch_dprimes' / set_name(meta)
batch_dprimes = jl.load(batch_dprime_file)

sites = set(batch_dprimes['dPCA']['dprime'].keys())

print('creating summary DataFrame')
df = list()
for site in sites:
    print(site)

    #### dpca
    sources = dict()
    sources['significance'] = dPCA_significance_dict[site]
    sources['dprime'], _ = cDP.flip_dprimes(dPCA_reals_dict[site], None, flip='max')
    t = times[:sources['dprime'].shape[-1]]
    for source, array in sources.items():

        # mean of transition pairs for each probe
        for pp, probe in enumerate(all_probes):
            mean = np.mean(array[pp,:,:],axis=0)
            popt, pvar = fts.exp_decay(t, mean, skip_error=True)
            parameters = dict()
            parameters['r0'] = popt[0]
            parameters['tau'] = -1/popt[1]
            parameters['max'] = np.max(mean)

            for parameter, value in parameters.items():
                d = {'siteid': site,
                     'cellid': np.nan,
                     'analysis': 'dPCA', # singel_cell, dPCA
                     'probe': f'probe_{probe}', # probe_n, mean
                     'transition_pair': 'mean', # t0_t1, mean
                     'parameter': parameter,
                     'source':source,
                     'value': value}
                df.append(d)

        # mean of probes for each transition pair
        for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
            mean = np.mean(array[:,tt,:],axis=0)
            popt, pvar = fts.exp_decay(t, mean, skip_error=True)
            parameters = dict()
            parameters['r0'] = popt[0]
            parameters['tau'] = -1/popt[1]
            parameters['max'] = np.max(mean)

            for parameter, value in parameters.items():
                d = {'siteid': site,
                     'cellid': np.nan,
                     'analysis': 'dPCA', # singel_cell, dPCA
                     'probe': 'mean', # probe_n or mean
                     'transition_pair': f'{trans[0]}_{trans[1]}', # t0_t1 or mean
                     'parameter': parameter,
                     'source':source,
                     'value': value}
                df.append(d)

        # full mean across probes and transition pairs
        mean = np.mean(array[:,:,:],axis=(0,1))
        popt, pvar = fts.exp_decay(t, mean, skip_error=True)
        parameters = dict()
        parameters['r0'] = popt[0]
        parameters['tau'] = -1/popt[1]
        parameters['max'] = np.max(mean)

        for parameter, value in parameters.items():
            d = {'siteid': site,
                 'cellid': np.nan,
                 'analysis': 'dPCA', # singel_cell, dPCA
                 'probe': 'mean', # probe_n or mean
                 'transition_pair': 'mean', # t0_t1 or mean
                 'parameter': parameter,
                 'source':source,
                 'value': value}
            df.append(d)

    #### single cell
    site_cells = SC_cells_array[SC_sites_array == site]
    for cell in site_cells:
        print(cell)
        sources = dict()
        sources['significance'] = SC_significance_dict[cell]
        sources['dprime'], _ = cDP.flip_dprimes(SC_reals_dict[cell], None, flip='max')
        t = times[:sources['dprime'].shape[-1]]
        for source, array in sources.items():

            # mean of transition pairs for each probe
            for pp, probe in enumerate(all_probes):
                mean = np.mean(array[pp,:,:],axis=0)
                popt, pvar = fts.exp_decay(t, mean, skip_error=True)
                parameters = dict()
                parameters['r0'] = popt[0]
                parameters['tau'] = -1/popt[1]
                parameters['max'] = np.max(mean)

                for parameter, value in parameters.items():
                    d = {'siteid': site,
                         'cellid': cell,
                         'analysis': 'single_cell', # singel_cell, dPCA
                         'probe': f'probe_{probe}', # probe_n, mean
                         'transition_pair': 'mean', # t0_t1, mean
                         'parameter': parameter,
                         'source':source,
                         'value': value}
                    df.append(d)

            # mean of probes for each transition pair
            for tt, trans in enumerate(itt.combinations(meta['transitions'], 2)):
                mean = np.mean(array[:,tt,:],axis=0)
                popt, pvar = fts.exp_decay(t, mean, skip_error=True)
                parameters = dict()
                parameters['r0'] = popt[0]
                parameters['tau'] = -1/popt[1]
                parameters['max'] = np.max(mean)

                for parameter, value in parameters.items():
                    d = {'siteid': site,
                         'cellid': cell,
                         'analysis': 'single_cell', # singel_cell, dPCA
                         'probe': 'mean', # probe_n or mean
                         'transition_pair': f'{trans[0]}_{trans[1]}', # t0_t1 or mean
                         'parameter': parameter,
                         'source':source,
                         'value': value}
                    df.append(d)

            # full mean across probes and transition pairs
            mean = np.mean(array[:,:,:],axis=(0,1))
            popt, pvar = fts.exp_decay(t, mean, skip_error=True)
            parameters = dict()
            parameters['r0'] = popt[0]
            parameters['tau'] = -1/popt[1]
            parameters['max'] = np.max(mean)

            for parameter, value in parameters.items():
                d = {'siteid': site,
                     'cellid': cell,
                     'analysis': 'single_cell', # singel_cell, dPCA
                     'probe': 'mean', # probe_n or mean
                     'transition_pair': 'mean', # t0_t1 or mean
                     'parameter': parameter,
                     'source':source,
                     'value': value}
                df.append(d)
DF = pd.DataFrame(df)
# add brain region
DF['region'] = [region_map[site] for site in DF.siteid]
del df

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / 'DF_summary' /set_name(meta)
if summary_DF_file.parent.exists() is False:
    summary_DF_file.parent.mkdir()
_ = jl.dump(DF, summary_DF_file)
