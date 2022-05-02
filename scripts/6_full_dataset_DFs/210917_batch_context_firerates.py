# from src.data.load import set_name
from src.root_path import  config_path
from src.data.rasters import load_site_formated_raster
from src.data.load import get_batch_ids

import itertools as itt
import numpy as np
import pandas as pd
from configparser import ConfigParser
import pathlib as pl
import joblib as jl
from tqdm import tqdm


"""
this version proceses the 10 sound permutation dataset.
Only works for permutations
mean 
"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 20,
        'zscore': True,
        'stim_type': 'permutations'}

# the old dataframe saved under this name (raster_fs == 30) is incorrect, as its pulling firing rates from the probe
# ctx_fr_DF_file = pl.Path(config['paths']['analysis_cache']) / f'210917_context_firing_rate_DF' / set_name(meta)
ctx_fr_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220429_ctx_fr_corrected_part_20hz'

# sites = set(get_site_ids(316).keys())
sites = set(get_batch_ids(316).siteid)
badsites = {'AMT031a', 'DRX008b', 'DRX021a', 'DRX023a', 'ley074a', 'TNC010a'}  # empirically decided
no_perm = {'ley058d'}  # sites without permutations
sites = sites.difference(badsites).difference(no_perm)


def firing_rates_to_DF(trialR, goodcells, meta):

    rep, chn, ctx, prb, tme = trialR.shape

    t = np.linspace(0, tme / meta['raster_fs'], tme,
                endpoint=False) * 1000
    R = np.mean(trialR, axis=0)
    DF = list()
    for (uu, unit), (cc, ctx), (pp, prb) in itt.product(
            enumerate(goodcells), enumerate(range(0, ctx)), enumerate(range(1, prb+1))):
        df = pd.DataFrame()
        df['time (ms)'] = t
        df['firing rate'] = R[uu, cc, pp, :]
        df['id'] = unit
        df['context'] = ctx
        df['probe'] = prb
        df['analysis'] = 'single cell'
        DF.append(df)
    DF = pd.concat(DF, ignore_index=True)
    return DF


# grows existing DF if any
recache=True
if ctx_fr_DF_file.exists() and not recache:
    DF = jl.load(ctx_fr_DF_file)
    ready_sites = set(DF.siteid.unique())
    sites = sites.difference(ready_sites)
    print('appening new sites to existing DF', sites)
    toconcat = [DF, ]
else:
    toconcat = list()

# loads data for site, organizes in DF and appends to growing DF
for site in tqdm(sites):
    trialR, goodcells = load_site_formated_raster(site, contexts='all', probes='all', meta=meta, part='context')
    toconcat.append(firing_rates_to_DF(trialR, goodcells, meta))


DF = pd.concat(toconcat, ignore_index=True)

DF.drop_duplicates(inplace=True)

if ctx_fr_DF_file.parent.exists() is False:
    ctx_fr_DF_file.parent.mkdir()
jl.dump(DF, ctx_fr_DF_file)

