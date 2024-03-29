import pathlib as pl
from configparser import ConfigParser

from src.root_path import config_path
from src.metrics.summaryDF import create_summary_DF
from src.utils.subsets import good_sites

"""
DEPRECATED: use instead 220804_singificant_abs_deltaFR


Refactored the DF script to be a function that takes all possible combinations of parameters to calculate and store
This instance calculates the amplitude as the integral of the cluster mass T-Score, for time intervals (chunks) of the 
probe response. The output of this script is mostly used on the regression analysis relating firing rate to the
usual contextual modulations metrics.
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
metrics = ['integral', 'integral_A', 'integral_B', 'integral_C', 'integral_D']
sources = ['real']
diff_metrics = ['delta_FR', ]
multiple_corrections = {'bf_cp': [1, 2],
                        # 'bf_ncp': [0, 1, 2],
                        }

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220719_chunked_amplitude_DF'
create_summary_DF(sites, loading_functions, cluster_thresholds, alpha, montecarlo, raster_meta, metrics, sources,
                  multiple_corrections, summary_DF_file, diff_metrics, keep_pvalues=False, recacheDF=True)

