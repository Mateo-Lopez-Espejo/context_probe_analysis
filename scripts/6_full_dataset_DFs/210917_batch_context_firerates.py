from src.data.load import set_name
from src.root_path import  config_path
from src.data.rasters import load_site_formated_raster

import itertools as itt
import numpy as np
import pandas as pd
from configparser import ConfigParser
import pathlib as pl
import joblib as jl


"""
this version proceses the 10 sound permutation dataset.
Only works for permutations
mean 
"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'zscore': True,
        'stim_type': 'permutations'}

ctx_fr_DF_file = pl.Path(config['paths']['analysis_cache']) / f'210917_context_firing_rate_DF' / set_name(meta)


expt = {'contexts': list(range(11)),
        'probes': list(range(1,11))}

# sites = set(get_site_ids(316).keys())
sites = {'TNC013a', 'TNC014a', 'TNC015a', 'TNC016a', 'TNC017a', 'TNC018a'}
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a' }  # empirically decided
sites = sites.difference(badsites)


def firing_rates_to_DF(trialR, goodcells, meta):
    t = np.linspace(0, trialR.shape[-1] / meta['raster_fs'], trialR.shape[-1],
                endpoint=False) * 1000
    R = np.mean(trialR, axis=0)
    DF = pd.DataFrame()
    for (uu, unit), (cc, ctx), (pp, prb) in itt.product(
            enumerate(goodcells), enumerate(expt['contexts']), enumerate(expt['probes'])):
        df = pd.DataFrame()
        df['time (ms)'] = t
        df['firing rate'] = R[uu, cc, pp, :]
        df['id'] = unit
        df['context'] = ctx
        df['probe'] = prb
        df['analysis'] = 'single cell'
        DF = DF.append(df, ignore_index=True)
    return DF


# grows existing DF if any
if ctx_fr_DF_file.exists():
    DF = pd.load(ctx_fr_DF_file)
    ready_sites = set(DF.siteid.unique())
    sites = sites.difference(ready_sites)
    print('appening new sites to existing DF', sites)
else:
    DF = pd.DataFrame()

# loads data for site, organizes in DF and appends to growing DF
for site in sites:
    trialR, _, goodcells = load_site_formated_raster(site, **expt, part='probe', meta=meta)
    DF = DF.append(firing_rates_to_DF(trialR, goodcells, meta),ignore_index=True)

DF.drop_duplicates(inplace=True)

if ctx_fr_DF_file.parent.exists() is False:
    ctx_fr_DF_file.parent.mkdir()
jl.dump(DF, ctx_fr_DF_file)

