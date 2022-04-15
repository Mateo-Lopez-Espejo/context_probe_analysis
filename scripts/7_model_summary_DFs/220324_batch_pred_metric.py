import itertools as itt
import pathlib as pl
from configparser import ConfigParser
import re

import numpy as np
import pandas as pd
import joblib as jl

from src.data.rasters import load_site_formated_prediction
from src.data.region_map import region_map
from src.metrics.time_series_summary import metrics_to_DF
from src.root_path import config_path
from src.utils.dataframes import add_classified_contexts
from src.models.modelnames import modelnames
from src.utils.subsets import cellid_subset_02, good_sites

"""
Quick and dirty calculation of context modulation metrics in model predictions
"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

# meta = {'reliability': 0.1,  # r value
#         'smoothing_window': 0,  # ms
#         'raster_fs': 30,
#         'montecarlo': 11000,
#         'zscore': True,
#         'stim_type': 'permutations'}

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220324_ctx_mod_metric_DF_pred'
summary_DF_file.parent.mkdir(parents=True, exist_ok=True)

metrics = ['mass_center', 'integral', 'mass_center_trunc', 'integral_trunc']

batch = 326

selected = {'STRF_long_relu', 'self_lone_relu', 'self_mod_relu', 'pop_lone_relu','pop_mod_relu'}
modelnames = {nickname:modelname for nickname, modelname in modelnames.items() if  nickname in selected}

recacheDF = False

if summary_DF_file.exists() and not recacheDF:
    DF = jl.load(summary_DF_file)
    ready_cells = set(DF.id.unique())
    cellids = cellid_subset_02.difference(ready_cells)
    print('appening new units to existing DF', cellids)
    to_concat = [DF,]
else:
    cellids = cellid_subset_02
    to_concat = list()

for cellid, (nickname, modelname) in itt.product(cellids, modelnames.items()):

    # just get one cellid for this example ToDo deleteme
    site = cellid.split('-')[0]
    raster, goodcells = load_site_formated_prediction(site, modelname=modelname, batch=batch, cellid=cellid)

    rep, chn, ctx, prb, tme = raster.shape
    ctx_pairs = list(itt.combinations(range(ctx), 2))

    # calculate difference on a single go
    pair_diff = np.empty((chn, len(ctx_pairs), prb, tme))
    for cpidx, (c0, c1) in enumerate(ctx_pairs):
        pair_diff[:, cpidx, :, :] = raster[0, :, c0, :, :] - raster[0, :, c1, :, :]

    # literal contexts and probe for dimlabdict
    contexts = list(range(0, pair_diff.shape[2] + 1))
    probes = list(range(1, pair_diff.shape[2] + 1))

    # creates label dictionalry
    fs = int(re.findall('\.fs\d*\.',modelname)[0][3:-1])
    dim_labl_dict = {'id': goodcells,
                     'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(contexts, 2)],
                     'probe': probes,
                     'time': np.linspace(0, pair_diff.shape[-1] / fs, pair_diff.shape[-1],
                                         endpoint=False) * 1000}

    # with empty mask for internal compatibility
    masked_dprime = np.ma.array(pair_diff, mask=np.full_like(pair_diff, False))
    df = metrics_to_DF(masked_dprime, dim_labl_dict, metrics=metrics)
    df['modelname'] = modelname
    df['nickname'] = nickname
    df['site'] = site
    df['region'] = region_map[site]
    df['stim_count'] = len(probes)

    to_concat.append(df)

DF = pd.concat(to_concat, ignore_index=True, axis=0)

# extra formatting
print(f'adding context clasification')
DF = add_classified_contexts(DF)

dups = np.sum(DF.duplicated().values)
if dups > 0:
    print(f'{dups} duplicated rows, what is wrong?, droping duplicates')
    DF.drop_duplicates(inplace=True)

print(DF.head(10))
jl.dump(DF, summary_DF_file)
