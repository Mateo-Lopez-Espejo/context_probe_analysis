import matplotlib.pyplot as plt
import numpy as np
import itertools as itt
import pathlib as pl

from cpn_load import load
from cpn_reliability import signal_reliability
import cpn_dPCA as cdPCA

from cpp_cache import make_cache, get_cache

import cpn_LDA as cLDA
import cpn_dprime as cDP
from progressbar import ProgressBar

from cpn_shuffle import shuffle_along_axis as shuffle
from scipy.stats import ranksums, wilcoxon
import pandas as pd
import seaborn as sn

import collections as col

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'significance': False,
        'montecarlo': 1000,
        'zscore': False}

analysis_name = 'windowed_correlations'
analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])


# for site, probe in zip(['AMT029a', 'ley070a'], [5, 2]):
#     # for site, probe in itt.product(all_sites, all_probes):

site = 'AMT029a'
probe = 5
part = 'probe'

# gets signal for hybridplot and toe select goodcellss
recs = load(site)

if len(recs) > 2:
    print(f'\n\n{recs.keys()}\n\n')

rec = recs['trip0']
sig = rec['resp']


# calculates response realiability and select only good cells to improve analysis
r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
goodcells = goodcells.tolist()

# get the full data raster Context x Probe x Rep x Neuron x Time
raster = cdPCA.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                               smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                               part=part, zscore=meta['zscore'])

# trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
trialR, _, _ = cdPCA.format_raster(raster)
trialR = trialR.squeeze()  # squeezes out probe

# calculates full LDA. i.e. considering all 4 categories
LDA_projection, LDA_weights = cLDA.fit_transform(trialR, 1)
dprime = cDP.pairwise_dprimes(LDA_projection.squeeze())