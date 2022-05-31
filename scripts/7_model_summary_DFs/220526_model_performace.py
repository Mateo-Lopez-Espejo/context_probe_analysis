import itertools as itt
import pathlib as pl
import re
from configparser import ConfigParser

import joblib as jl
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.region_map import region_map
from src.metrics.time_series_summary import metrics_to_DF
from src.models.modelnames import modelnames
from src.models.param_tools import get_pred_err
from src.root_path import config_path
from src.utils.subsets import cellid_A1_fit_set, cellid_PEG_fit_set, batch_map

"""
pools all the model performance data for all neurons
"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220526_resp_pred_metrics_by_chunks'
summary_DF_file.parent.mkdir(parents=True, exist_ok=True)


selected = {'matchl_STRF', 'matchl_self', 'matchl_pop', 'matchl_full'}

modelnames = {nickname: modelname for nickname, modelname in modelnames.items() if nickname in selected}