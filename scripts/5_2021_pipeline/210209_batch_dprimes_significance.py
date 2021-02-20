from src.metrics.significance import _significance
from src.metrics.consolidated_dprimes import single_cell_dprimes, probewise_LDA_dprimes, probewise_dPCA_dprimes, full_dPCA_dprimes
from src.metrics.dprime import flip_dprimes
from src.data.load import get_site_ids
from src.metrics.consolidated_metrics import metrics_to_DF

import itertools as itt
import numpy as np
import pandas as pd

rec_recache = False
dprime_recache = False

signif_tails = 'both'
alpha=0.01


meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None}

analysis_functions = {'SC': single_cell_dprimes,'LDA':probewise_LDA_dprimes,
                      'pdPCA': probewise_dPCA_dprimes, 'fdPCA': full_dPCA_dprimes}


permutations = {'contexts': [0, 1, 2, 3, 4],
                'probes': [1, 2, 3, 4],
                'stim_type': 'permutations'}

triplets = {'contexts': ['silence', 'continuous', 'similar', 'sharp'],
            'probes':[2, 3, 5, 6],
            'stim_type': 'triplets'}

experiments = [permutations, triplets]

multiple_corrections = {'none':None,
                        'full': [1,2,3],
                        'time': [3],
                        'probe': [2,3],
                        'context_pair': [1,3]}

metrics = ['significant_abs_mass_center', 'significant_abs_sum']


sites = set(get_site_ids(316).keys())
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a' }  # empirically decided
sites = sites.difference(badsites)
sites = ['CRD004a']

DF = pd.DataFrame()
bads = list()
for site, expt, (fname, func) in itt.product(sites, experiments, analysis_functions.items()):
    print(site, expt['stim_type'], fname)
    expt = expt.copy()
    meta['stim_type'] = expt.pop('stim_type')
    try:
        dprime, shuffled_dprime, goodcells = func(site, **expt, meta=meta)
    except:
        bads.append((site, expt['stim_type'], fname))

    if fname != 'SC':
        chan_name = [np.nan]
    else:
        chan_name = goodcells


    # calculats different significaces/corrections
    # calculate significant time bins, both raw and corrected for multiple comparisons

    for corr_name, corr in multiple_corrections.items():

        significance, confidence_interval = _significance(dprime, shuffled_dprime, corr,
                                                                            alpha=alpha)
        fliped, _ = flip_dprimes(dprime, None, flip='sum')

        # masks dprime with different significances
        masked = np.where(significance, fliped, 0)

        # creates label dictionalry
        dim_lab_dict = {'cellid': chan_name,
                        'context_pair': [f'{c1}_{c2}' for c1, c2 in itt.combinations(expt['contexts'], 2)],
                        'probe': expt['probes'],
                        'time': np.linspace(0, dprime.shape[-1] / meta['raster_fs'], dprime.shape[-1],
                                            endpoint=False) * 1000}

        df = metrics_to_DF(masked, dim_lab_dict, metrics=metrics)
        df['mult_comp_corr'] = corr_name
        df['stim_type'] = meta['stim_type']
        df['analysis'] = fname
        df['site'] = site

        DF = DF.append(df)

