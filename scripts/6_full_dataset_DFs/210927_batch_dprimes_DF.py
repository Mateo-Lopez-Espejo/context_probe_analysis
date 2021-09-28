from src.metrics.significance import _significance
from src.metrics.consolidated_dprimes import single_cell_dprimes, probewise_LDA_dprimes, probewise_dPCA_dprimes, full_dPCA_dprimes
from src.metrics.dprime import flip_dprimes
from src.data.load import set_name
from src.root_path import  config_path

import itertools as itt
import numpy as np
import pandas as pd
from configparser import ConfigParser
import pathlib as pl
from joblib import dump, load

"""
here I am trying to orgamized raw dprime values instead of their summary metrics in a dataframe, this whith the purpose 
of doing a regression against the difference in the contexts firing rate.

"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))


signif_tails = 'both'
alpha=0.05

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None,
        'stim_type': 'permutations'}

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'210927_dprime_full_DF_alpha_{alpha}' / set_name(meta)

analysis_functions = {'SC': single_cell_dprimes, #'LDA':probewise_LDA_dprimes,
                      'fdPCA': full_dPCA_dprimes} #,'pdPCA': probewise_dPCA_dprimes}

expt = {'contexts': list(range(11)),
        'probes': list(range(1,11))}

multiple_corrections = {'consecutive_4': ([3], 4)}

metrics = ['significant_abs_mass_center', 'significant_abs_sum']

# sites = set(get_site_ids(316).keys())
sites = set(['TNC013a', 'TNC014a', 'TNC015a', 'TNC016a', 'TNC017a', 'TNC018a'])
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a' }  # empirically decided
sites = sites.difference(badsites)


if summary_DF_file.exists():
    DF = load(summary_DF_file)
    ready_sites = set(DF.siteid.unique())
    sites = sites.difference(ready_sites)
    print('appening new sites to existing DF', sites)
else:
    DF = pd.DataFrame()

bads = list()
for site, (fname, func) in itt.product(sites, analysis_functions.items()):
    print(site)

    dprime, shuff_dprime_quantiles, goodcells, var_capt = func(site, **expt, meta=meta)

    # for analysis with dimensionality reduction, changes the cellname to nan for proper dimension labeling.
    if fname != 'SC':
        chan_name = [np.nan]
    else:
        chan_name = goodcells[0][:7]

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


        dff = pd.DataFrame()
        for (uu, unit), (cc, ctx), (pp, prb) in itt.product(
                enumerate(chan_name), enumerate(dim_labl_dict['context_pair']), enumerate(expt['probes'])):

            # to avoid superfluous data,  keeps only significant dprime slices
            this_dprime = fliped[uu, cc, pp, :]
            this_signif = significance[uu, cc, pp, :]

            if np.any(this_signif):
                pass
            else:
                continue

            df = pd.DataFrame()
            df['time (ms)'] = dim_labl_dict['time']
            df['dprime'] = this_dprime
            df['significant'] = this_signif
            df['id'] = unit
            df['context_pair'] = ctx
            df['probe'] = prb
            df['analysis'] = fname
            dff = dff.append(df, ignore_index=True)

        DF = DF.append(dff,ignore_index=True)

print('failed sites: ', bads)

DF.drop_duplicates(inplace=True)

if summary_DF_file.parent.exists() is False:
    summary_DF_file.parent.mkdir()
dump(DF, summary_DF_file)

