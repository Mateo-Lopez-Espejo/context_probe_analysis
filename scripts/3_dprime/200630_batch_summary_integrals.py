from configparser import ConfigParser
import pathlib as pl
import joblib as jl

import numpy as np
import pandas as pd
import scipy.stats as sst

import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation

import fancy_plots as fplt
from cpp_cache import set_name

"""
2020-06-30
Previously I used a fitted exponential decay to describe the evolution of contextual effects over time, however the fitting
in many cases was unadequate, adding artifactual outliers. An alterlative to this fitting approach is to instead consider
a fitless alternative, like the integral of the contextual effect. Here I explore two alternatives of this
1. total integral of the dprime time series 
2. integral of the dprime series only at time bins with significant dprime values
"""

config = ConfigParser()
if pl.Path('../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../context_probe_analysis/config/settings.ini'))
elif pl.Path('../../../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../../../context_probe_analysis/config/settings.ini'))
else:
    raise FileNotFoundError('config file coluld not be foud')

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None}

# transferable plotting parameters
plt.rcParams['svg.fonttype'] = 'none'
sup_title_size = 30
sub_title_size = 20
ax_lab_size = 15
ax_val_size = 11
full_screen = [19.2, 9.83]
sns.set_style("ticks")

########################################################################################################################
########################################################################################################################
# data frame containing all the important summary data, i.e. exponential decay fits for dprime and significance, for
# all combinations of transition pairs, and probes,  for the means across probes, transitions pairs or for both, and
# for the single cell analysis or the dPCA projections
summary_DF_file = pl.Path(config['paths']['analysis_cache']) / 'DF_summary' / set_name(meta)
print('loading cached summary DataFrame')
DF = jl.load(summary_DF_file)










