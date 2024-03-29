import pathlib as pl
from configparser import ConfigParser

from src.root_path import config_path
from src.metrics.summaryDF import create_summary_DF
from src.utils.subsets import good_sites

"""
in this script we use the T-score cluster mass significance test to define significant time bins, and then calculate
the absolute integral (a measure of amplitude) for the delta firing rate, see diff_metrics = ['delta_FR', ],
This is the first script implementing and using this new feature, which simplifies the units of the metrics we are displaying
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
metrics = ['integral', 'integral_A', 'integral_B', 'integral_C', 'integral_D', 'last_bin']
sources = ['real', ]
diff_metrics = ['delta_FR', ]
multiple_corrections = {'bf_cp': [1, 2]}

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220804_significant_abs_deltaFR_DF'

create_summary_DF(sites, loading_functions, cluster_thresholds, alpha, montecarlo, raster_meta, metrics, sources,
                  multiple_corrections, summary_DF_file, recacheDF=True, diff_metrics=diff_metrics)

