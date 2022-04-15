import itertools as itt
import pathlib as pl
from configparser import ConfigParser
import re

import scipy.stats as sst

import numpy as np
import pandas as pd
import joblib as jl

from src.root_path import config_path
from src.data.region_map import region_map
from src.metrics.time_series_summary import metrics_to_DF
from src.models.modelnames import modelnames
from src.models.param_tools import get_pred_err
from src.utils.dataframes import ndim_array_to_long_DF
from src.utils.subsets import cellid_A1_fit_set, cellid_PEG_fit_set, batch_map

"""
Quick and dirty calculation of context modulation metrics in model predictions, using contextual differece
"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220412_cordoeff_by_bins'
summary_DF_file.parent.mkdir(parents=True, exist_ok=True)

metrics = ['integral']

#  ctxpr, prb, tme
big_time_bins = {'A': np.s_[:, :, :25],
                 'B': np.s_[:, :, 25:50],
                 'C': np.s_[:, :, 50:75],
                 'D': np.s_[:, :, 75:],
                 'full': np.s_[...]}

selected = {'STRF_long_relu', 'pop_lone_relu', 'pop_mod_relu'}
modelnames = {nickname: modelname for nickname, modelname in modelnames.items() if nickname in selected}

recacheDF = False
all_cellids = cellid_A1_fit_set.union(cellid_PEG_fit_set)

if summary_DF_file.exists() and not recacheDF:
    DF = jl.load(summary_DF_file)
    ready_cells = set(DF.id.unique())
    # ready_mdls_cells = DF.loc[:,['id','nickname']]
    cellids = all_cellids.difference(ready_cells)
    print('appening new units to existing DF', cellids)
    to_concat = [DF, ]
else:
    cellids = all_cellids
    to_concat = list()

unfitted = list()
for cellid in cellids:
    site = cellid.split('-')[0]
    for mm, (nickname, modelname) in enumerate(modelnames.items()):
        # loads the recorded and predicted pairwise differences
        _, _, diff_resp, diff_pred = get_pred_err(cellid, batch_map[cellid], modelname, part='probe',
                                                  retur_diffs=True)

        ctx_pr, prb, tme = diff_resp.shape

        # literal contexts and probe for dimlabdict
        contexts = list(range(0, prb + 1))
        probes = list(range(1, prb + 1))

        for bin_name, slicer in big_time_bins.items():

            diff_resp_chunk = diff_resp[slicer][np.newaxis, ...]
            diff_pred_chunk = diff_pred[slicer][np.newaxis, ...]

            ##### first simple integtral over the difference #####
            # creates label dictionalry
            fs = int(re.findall('\.fs\d*\.', modelname)[0][3:-1])
            dim_labl_dict = {'id': [cellid],
                             'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(contexts, 2)],
                             'probe': probes,
                             'time': np.linspace(0, diff_pred_chunk.shape[-1] / fs, diff_pred_chunk.shape[-1],
                                                 endpoint=False) * 1000}  # todo, ensure to shift time depending on big bin

            # with empty mask for internal compatibility
            masked_dprime = np.ma.array(diff_pred_chunk, mask=np.full_like(diff_pred_chunk, False))
            df = metrics_to_DF(masked_dprime, dim_labl_dict, metrics=metrics)
            df['modelname'] = modelname
            df['nickname'] = nickname
            df['site'] = site
            df['region'] = region_map[site]
            df['stim_count'] = len(probes)
            df['time_bin'] = bin_name
            to_concat.append(df)

            # also saves the response, but only for the first, model since it should be equal for all models
            if mm == 0:
                masked_dprime = np.ma.array(diff_resp_chunk, mask=np.full_like(diff_resp_chunk, False))
                df = metrics_to_DF(masked_dprime, dim_labl_dict, metrics=metrics)
                df['modelname'] = 'response'
                df['nickname'] = 'response'
                df['site'] = site
                df['region'] = region_map[site]
                df['stim_count'] = len(probes)
                df['time_bin'] = bin_name
                to_concat.append(df)

            ##### second. correlation coefficient #####
            chn, ctx_pr, prb, tme = diff_resp_chunk.shape
            r_val_array = np.empty((chn, ctx_pr, prb))
            for n, c, p in np.ndindex(r_val_array.shape):
                r_val_array[n, c, p] = sst.pearsonr(diff_resp_chunk[n, c, p, :], diff_pred_chunk[n, c, p, :])[0]

            dim_labl_dict = {'id': [cellid],
                             'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(contexts, 2)],
                             'probe': probes}

            df = ndim_array_to_long_DF(r_val_array, dim_labl_dict)
            df['modelname'] = modelname
            df['nickname'] = nickname
            df['site'] = site
            df['region'] = region_map[site]
            df['stim_count'] = len(probes)
            df['time_bin'] = bin_name
            df['metric'] = 'pearsons-r'
            to_concat.append(df)

DF = pd.concat(to_concat, ignore_index=True, axis=0)

# extra formatting
# print(f'adding context clasification')
# DF = add_classified_contexts(DF)

dups = np.sum(DF.duplicated().values)
if dups > 0:
    print(f'{dups} duplicated rows, what is wrong?, droping duplicates')
    DF.drop_duplicates(inplace=True)

print(DF.head(10), DF.shape)
jl.dump(DF, summary_DF_file)
