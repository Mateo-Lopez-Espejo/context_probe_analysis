import itertools as itt
import pathlib as pl
from configparser import ConfigParser
import re
from tqdm import tqdm

import scipy.stats as sst
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
import joblib as jl

from src.root_path import config_path
from src.data.region_map import region_map
from src.metrics.time_series_summary import metrics_to_DF
from src.models.modelnames import modelnames
from src.models.param_tools import get_pred_err
from src.utils.dataframes import ndim_array_to_long_DF
from src.utils.subsets import cellid_A1_fit_set, cellid_PEG_fit_set, batch_map, cellid_subset_02, cellid_subset_01

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
                 'D': np.s_[:, :, 75:],}
                 # 'full': np.s_[...]}

selected = {'STRF_long_relu', 'pop_lone_relu', 'pop_mod_relu', 'self_mod_relu', 'self_lone_relu', 'STP_STRF1_relu',
            'STP_STRF2_relu'}
modelnames = {nickname: modelname for nickname, modelname in modelnames.items() if nickname in selected}

recacheDF = False
all_cellids = cellid_A1_fit_set.union(cellid_PEG_fit_set)


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
    print(f'appening new cellid/model combinations to existing DF, n= {total_iter}',)
    to_concat = [DF, ]
else:
    def cells_models_todo():
        return itt.product(all_cellids, modelnames.keys())

    cellids_todo = all_cellids
    to_concat = list()


not_fitted = list()

# for cellid in cellids_todo:
#     for mm, (nickname, modelname) in enumerate(modelnames.items()):

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

            # goodness of fit metrics, todo do I need these values? they are big and time consuming
            # chn, ctx_pr, prb, tme = diff_resp_chunk.shape
            #
            # ##### second. correlation coefficient #####
            # r_val_array = np.empty((chn, ctx_pr, prb))
            # for n, c, p in np.ndindex(r_val_array.shape):
            #     r_val_array[n, c, p] = sst.pearsonr(diff_resp_chunk[n, c, p, :], diff_pred_chunk[n, c, p, :])[0]
            #
            # dim_labl_dict = {'id': [cellid],
            #                  'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(contexts, 2)],
            #                  'probe': probes}
            #
            # df = ndim_array_to_long_DF(r_val_array, dim_labl_dict)
            # df['modelname'] = modelname
            # df['nickname'] = nickname
            # df['site'] = site
            # df['region'] = region_map[site]
            # df['stim_count'] = len(probes)
            # df['time_bin'] = bin_name
            # df['metric'] = 'pearsons-r'
            # to_concat.append(df)


            # #### third. determination coeffeicient ####
            # chn, ctx_pr, prb, tme = diff_resp_chunk.shape
            # R2_val_array = np.empty((chn, ctx_pr, prb))
            # for n, c, p in np.ndindex(R2_val_array.shape):
            #     R2_val_array[n, c, p] = r2_score(diff_resp_chunk[n, c, p, :], diff_pred_chunk[n, c, p, :])
            #
            # dim_labl_dict = {'id': [cellid],
            #                  'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(contexts, 2)],
            #                  'probe': probes}
            #
            # df = ndim_array_to_long_DF(R2_val_array, dim_labl_dict)
            # df['modelname'] = modelname
            # df['nickname'] = nickname
            # df['site'] = site
            # df['region'] = region_map[site]
            # df['stim_count'] = len(probes)
            # df['time_bin'] = bin_name
            # df['metric'] = 'R2'
            # to_concat.append(df)



# for ease of adding new models, now adds the actual response values for a single model
# just need a list of neurons to cache!
not_models = list()
for cellid in tqdm(cellids_todo, total=len(cellids_todo)):
    site = cellid.split('-')[0]

    for nickname, modelname in modelnames.items():
        try:
            _, _, diff_resp, _ = get_pred_err(cellid, batch_map[cellid], modelname, part='probe',
                                                      retur_diffs=True)
            break
        except:
            not_fitted.append((cellid, nickname))
            continue
    else:
        not_models.append(cellid)

    ctx_pr, prb, tme = diff_resp.shape

    # literal contexts and probe for dimlabdict
    contexts = list(range(0, prb + 1))
    probes = list(range(1, prb + 1))

    for bin_name, slicer in big_time_bins.items():

        diff_resp_chunk = diff_resp[slicer][np.newaxis, ...]

        ##### first simple integtral over the difference #####
        # creates label dictionalry
        fs = int(re.findall('\.fs\d*\.', modelname)[0][3:-1])
        dim_labl_dict = {'id': [cellid],
                         'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(contexts, 2)],
                         'probe': probes,
                         'time': np.linspace(0, diff_pred_chunk.shape[-1] / fs, diff_pred_chunk.shape[-1],
                                             endpoint=False) * 1000}  # todo, ensure to shift time depending on big bin

        masked_dprime = np.ma.array(diff_resp_chunk, mask=np.full_like(diff_resp_chunk, False))
        df = metrics_to_DF(masked_dprime, dim_labl_dict, metrics=metrics)
        df['modelname'] = 'response'
        df['nickname'] = 'response'
        df['site'] = site
        df['region'] = region_map[site]
        df['stim_count'] = len(probes)
        df['time_bin'] = bin_name
        to_concat.append(df)

DF = pd.concat(to_concat, ignore_index=True, axis=0)

print(f'###### {len(not_fitted)} not fitted: ######\n', not_fitted)
print(f'###### {len(not_models)} lacks models: ######\n', not_models)

# extra formatting
# print(f'adding context clasification')
# DF = add_classified_contexts(DF)

dups = np.sum(DF.duplicated().values)
if dups > 0:
    print(f'{dups} duplicated rows, what is wrong?, droping duplicates')
    DF.drop_duplicates(inplace=True)

print(DF.head(10), DF.shape)
jl.dump(DF, summary_DF_file)
