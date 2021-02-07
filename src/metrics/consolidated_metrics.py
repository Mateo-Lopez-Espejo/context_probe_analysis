import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import joblib as jl
import numpy as np
import pandas as pd
from scipy.integrate import trapz

from src.metrics import dprime as cDP
from src.utils import fits as fts
from src.data.cache import set_name
from src.data.region_map import region_map

"""
takes all the dprimes and pvalues, fits exponential decays to both the dprimes and the profiles of dprime
significance (derived from pvalues). 
This is done for all combinations of probe, and context transtion pairs.
This is done for single cells (SC), probewise dPCA (pdPCA) and full_dPCA (fdPCA).
"""

config = ConfigParser()
config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': [0, 1, 2, 3, 4],
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None}

recache = True

# loads the raw calculated dprimes and montecarlos
batch_dprime_file = pl.Path(config['paths']['analysis_cache']) / 'prm_dprimes_v2' / set_name(meta)
batch_dprimes = jl.load(batch_dprime_file)

# defines significant values based on loaded pvalue and defined threshold
threshold = 0.01
for analysis_name, mid_dict in batch_dprimes.items():
    mid_dict['significance'] = {key: (val <= threshold) for key, val in mid_dict['pvalue'].items()}

# set up the time bin labels in milliseconds, this is critical for plotting and calculating the tau
nbin = np.max([value.shape[-1] for value in batch_dprimes['SC']['dprime'].values()])
times = np.linspace(0, nbin / meta['raster_fs'], nbin, endpoint=False) * 1000

sites = set(batch_dprimes['pdPCA']['dprime'].keys())
all_probes = [1, 2, 3, 4]
all_trans = [f'{t0}_{t1}' for t0, t1 in itt.combinations(meta['transitions'], 2)]

# creates and caches, or loads the DF

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / 'prm_summary_DF_v2' / set_name(meta)