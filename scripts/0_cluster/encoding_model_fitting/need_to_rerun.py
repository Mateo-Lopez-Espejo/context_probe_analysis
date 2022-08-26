from src.data.load import get_CPN_ids, get_runclass_ids, get_batch_ids
from src.utils.subsets import bad_sites, cellid_A1_fit_set, cellid_PEG_fit_set
import pandas as pd
import joblib as jl
import pathlib as pl
from configparser import ConfigParser
from src.root_path import config_path
from nems.db import batch_comp

from src.models.modelnames import modelnames as all_modelnames
from src.data.region_map import region_map

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
model_df_file = pl.Path(config['paths']['analysis_cache']) / f'220412_resp_pred_metrics_by_chunks'

DF = jl.load(model_df_file)

proc_cells = set(DF.id.unique())

all_to_fit = cellid_A1_fit_set.union(cellid_PEG_fit_set)

to_reprocess = all_to_fit.difference(proc_cells)

print(to_reprocess)
