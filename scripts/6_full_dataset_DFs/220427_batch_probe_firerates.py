import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import joblib as jl
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.load import get_site_ids
from src.data.rasters import load_site_formated_raster
from src.root_path import config_path

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

ctx_fr_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220427_probe_firerates'
ctx_fr_DF_file.parent.mkdir(parents=True, exist_ok=True)

sites = set(get_site_ids(316).keys())
badsites = {'AMT031a', 'DRX008b', 'DRX021a', 'DRX023a', 'ley074a', 'TNC010a'}  # empirically decided
no_perm = {'ley058d'}  # sites without permutations
sites = sites.difference(badsites).difference(no_perm)
print(f'all sites: \n{sites}\n')



recacheDF = True
if ctx_fr_DF_file.exists() and not recacheDF:
    DF = jl.load(ctx_fr_DF_file)
    ready_sites = set(DF.site.unique())
    sites = sites.difference(ready_sites)
    print('appening new sites to existing DF', sites)
    to_concat = [DF, ]
else:
    to_concat = list()



for site in tqdm(sites):

    trialR, goodcells = load_site_formated_raster(site, contexts='all', probes='all', raster_fs=meta['raster_fs'])

    rep, chn, ctx, prb, tme = trialR.shape

    # take the mean across repetitions, contexts and time
    R = np.mean(trialR[:,:,:,:,20:], axis=(0, 2, 4))  # shape chn x prb

    df = list()
    for (uu, unit), (pp, prb) in itt.product(enumerate(goodcells), enumerate(range(1, prb + 1))):
        d = {'id': unit,
             'site': unit.split('-')[0],
             'probe': prb,
             'metric': 'mean_fr',
             'value': R[uu,pp]}
        df.append(d)

    to_concat.append(pd.DataFrame(df))

DF = pd.concat(to_concat, ignore_index=True, axis=0)

dups = np.sum(DF.duplicated().values)
if dups > 0:
    print(f'{dups} duplicated rows, what is wrong?, droping duplicates')
    DF.drop_duplicates(inplace=True)

print(DF.head(10))
jl.dump(DF, ctx_fr_DF_file)
