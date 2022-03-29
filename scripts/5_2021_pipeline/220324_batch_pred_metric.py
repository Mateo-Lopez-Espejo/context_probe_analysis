import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import numpy as np
import pandas as pd
import joblib as jl

from src.data.load import get_site_ids
from src.data.rasters import load_site_formated_prediction
from src.data.region_map import region_map
from src.metrics.time_series_summary import metrics_to_DF
from src.root_path import config_path
from src.utils.dataframes import add_classified_contexts
from src.models.modelnames import modelnames

"""

"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 11000,
        'zscore': True,
        'stim_type': 'permutations'}

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220324_ctx_mod_metric_DF_pred'
summary_DF_file.parent.mkdir(parents=True, exist_ok=True)


metrics = ['mass_center', 'integral']

sites = set(get_site_ids(316).keys())
badsites = {'AMT031a', 'DRX008b', 'DRX021a', 'DRX023a', 'ley074a', 'TNC010a'}  # empirically decided
no_perm = {'ley058d'}  # sites without permutations
sites = sites.difference(badsites).difference(no_perm)


sites = ('TNC014a', )
batch = 326

selected = ['STRF_long', ]
modelnames = {nickname:modelname for nickname, modelname in modelnames.items() if  nickname in selected}

fs = 100 # todo extrac it dinamically from modelname??

print(f'all sites: \n{sites}\n')


recacheDF = True

if summary_DF_file.exists() and not recacheDF:
    DF = jl.load(summary_DF_file)
    ready_sites = set(DF.siteid.unique())
    sites = sites.difference(ready_sites)
    print('appening new sites to existing DF', sites)
    to_concat = [DF,]
else:
    to_concat = list()

for site, (nickname, modelname) in itt.product(sites, modelnames.items()):

    raster, goodcells = load_site_formated_prediction(site, modelspec=modelname, batch=batch)

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
    dim_labl_dict = {'id': goodcells,
                     'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(contexts, 2)],
                     'probe': probes,
                     'time': np.linspace(0, pair_diff.shape[-1] / fs, pair_diff.shape[-1],
                                         endpoint=False) * 1000}

    # with empty mask for internal compatibility
    masked_dprime = np.ma.array(pair_diff, mask=np.full_like(pair_diff, False))
    df = metrics_to_DF(masked_dprime, dim_labl_dict, metrics=metrics)
    df['modelname'] = modelname
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
