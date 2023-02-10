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
calculates metrics (integral and center of mass) on the delta firing rate of model predictions and real responses.
This in lieu of the more complicated cluster mass analyssi which requieres multiple trials (unavailable for model
predictions) 
"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220412_resp_pred_metrics_by_chunks_bak' # old bakcup
summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220412_resp_pred_metrics_by_chunks' # refreshed neuron names/models

summary_DF_file.parent.mkdir(parents=True, exist_ok=True)

metrics = ['integral', 'mass_center', 'integral_trunc1.5', 'mass_center_trunc1.5']

#  ctxpr, prb, tme
# asumes modelse with fr 100 and duration 1 second. assersion appear late
time_chunks = {'A': np.s_[:, :, :25],
                 'B': np.s_[:, :, 25:50],
                 'C': np.s_[:, :, 50:75],
                 'D': np.s_[:, :, 75:],
                 'full': np.s_[:, :, :]}

selected = {'matchl_STRF', 'matchl_self', 'matchl_pop', 'matchl_full'}

modelnames = {nickname: modelname for nickname, modelname in modelnames.items() if nickname in selected}

all_cellids = cellid_A1_fit_set.union(cellid_PEG_fit_set)
# all_cellids = {'TNC014a-22-2'}

recacheDF = True
if summary_DF_file.exists() and not recacheDF:
    DF = jl.load(summary_DF_file)
    DF_done = DF.loc[:, ['id', 'nickname']].drop_duplicates()


    def cells_models_todo():
        for cellid, nickname in itt.product(all_cellids, modelnames.keys()):
            if DF_done.query(f"id == '{cellid}' and nickname == '{nickname}' and nickname != 'respose'").size != 0:
                continue
            else:
                yield cellid, nickname


    ready_cells = set(DF.id.unique())
    cellids_todo = all_cellids.difference(ready_cells)
    total_iter = len(list(cells_models_todo()))
    print(f'appening new cellid/model combinations to existing DF, n= {total_iter}', )
    to_concat = [DF, ]
else:
    def cells_models_todo():
        return itt.product(all_cellids, modelnames.keys())


    total_iter = len(all_cellids) * len(modelnames)
    cellids_todo = all_cellids
    to_concat = list()

not_fitted = list()

for cellid, nickname in tqdm(cells_models_todo(), total=total_iter):
    site = cellid.split('-')[0]
    modelname = modelnames[nickname]
    # loads the recorded and predicted pairwise differences
    try:
        _, _, diff_resp, diff_pred = get_pred_err(cellid, batch_map[cellid], modelname, part='probe',
                                                  retur_diffs=True)
    except:
        not_fitted.append((cellid, nickname))
        continue

    ctx_pr, prb, tme = diff_resp.shape

    # literal contexts and probe for dimlabdict
    contexts = list(range(0, prb + 1))
    probes = list(range(1, prb + 1))

    # define time stamps for the whole probe predictionn
    fs = int(re.findall('\.fs\d*\.', modelname)[0][3:-1])
    assert fs == 100

    time = np.linspace(0, diff_resp.shape[-1] / fs, diff_resp.shape[-1],
                       endpoint=False) * 1000

    for chunk_name, slicer in time_chunks.items():
        diff_resp_chunk = diff_resp[slicer][np.newaxis, ...]
        diff_pred_chunk = diff_pred[slicer][np.newaxis, ...]

        ##### first simple integtral over the difference #####
        # creates label dictionalry
        dim_labl_dict = {'id': [cellid],
                         'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(contexts, 2)],
                         'probe': probes,
                         'time': time[slicer[-1]]}  # todo, ensure to shift time depending on big bin

        # with empty mask, i.e. no mask for internal compatibility
        masked_resp = np.ma.array(diff_resp_chunk, mask=np.full_like(diff_resp_chunk, False))
        masked_pred = np.ma.array(diff_pred_chunk, mask=np.full_like(diff_pred_chunk, False))
        resp_df = metrics_to_DF(masked_resp, dim_labl_dict, metrics=metrics)
        df = metrics_to_DF(masked_pred, dim_labl_dict, metrics=metrics)
        df['resp'] = resp_df['value']  # keeps track of the original response delta fr
        df['modelname'] = modelname
        df['nickname'] = nickname
        df['site'] = site
        df['region'] = region_map[site]
        df['stim_count'] = len(probes)
        df['chunk'] = chunk_name
        to_concat.append(df)

DF = pd.concat(to_concat, ignore_index=True, axis=0)

print(f'###### {len(not_fitted)} not fitted: ######\n', not_fitted)

for col in  ['id', 'context_pair', 'probe', 'metric', 'modelname', 'nickname', 'site',
             'region', 'stim_count', 'chunk']:
    DF[col] = DF[col].astype('category')

for col in ['value', 'resp']:
    DF[col] = pd.to_numeric(DF[col], downcast='float')

dups = np.sum(DF.duplicated().values)
if dups > 0:
    print(f'{dups} duplicated rows, what is wrong?, droping duplicates')
    DF.drop_duplicates(inplace=True)

print(DF.head(10), DF.shape)
jl.dump(DF, summary_DF_file)
