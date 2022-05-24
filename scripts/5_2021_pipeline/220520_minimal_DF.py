import pathlib as pl
from configparser import ConfigParser

from src.root_path import config_path
from src.metrics.summaryDF import create_summary_DF
from src.utils.subsets import good_sites

"""
Refactored the DF script to be a function that takes all possible combinations of parameters to calculate and store
This One specific instance is only holding the bare minimum metrics of real tscore clusters-mass 
"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))


# Kinda baseline minimal parameterse

sites = good_sites
loading_functions = ['SC', 'PCA']
cluster_thresholds = [0.05, ]
alpha = 0.05
montecarlo = 11000
raster_meta = {'reliability': 0.1,  # r value
               'smoothing_window': 0,  # ms
               'raster_fs': 20,
               'zscore': True,
               'stim_type': 'permutations'}
metrics = ['integral', 'last_bin']
sources = ['real']
multiple_corrections = {'bf_cp': [1, 2], 'bf_ncp': [0, 1, 2], }

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220520_minimal_DF'
create_summary_DF(sites, loading_functions, cluster_thresholds, alpha, montecarlo, raster_meta, metrics, sources,
                  multiple_corrections, summary_DF_file, recacheDF=True)

