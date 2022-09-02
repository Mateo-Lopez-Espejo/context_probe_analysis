import pathlib as pl
from configparser import ConfigParser

from src.root_path import config_path
from src.metrics.summaryDF import create_summary_DF
from src.utils.subsets import good_sites

"""
Following the simplification of the amplitude metric ( the absolute integral of the delta firign rate on significant time bins)
here we just calculate the absolute delta firing rate disregarding significance, for the quarters of time.

note the "nosig" tag on the metric names. 

I am keeping the significance for the whole integral calculation, as i will use it to filter instances that show any
significant effects
"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))


# Kinda baseline minimal parameterse

sites = good_sites
loading_functions = ['SC', ]
cluster_thresholds = [0.05, ]
alpha = 0.05
montecarlo = 11000
raster_meta = {'reliability': 0.1,  # r value
               'smoothing_window': 0,  # ms
               'raster_fs': 20,
               'zscore': True,
               'stim_type': 'permutations'}
metrics = ['integral', 'integral_nosig', 'last_bin',
           'integral_nosig_A', 'integral_nosig_B', 'integral_nosig_C', 'integral_nosig_D']
sources = ['real', ]
diff_metrics = ['delta_FR', ]
multiple_corrections = {'bf_cp': [1, 2]}

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220818_abs_deltaFR_DF'

create_summary_DF(sites, loading_functions, cluster_thresholds, alpha, montecarlo, raster_meta, metrics, sources,
                  multiple_corrections, summary_DF_file, recacheDF=True, diff_metrics=diff_metrics,
                  keep_pvalues=False)

