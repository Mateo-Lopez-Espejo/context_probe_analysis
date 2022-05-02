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

fr_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220427_probe_firerates'
fr_DF_file.parent.mkdir(parents=True, exist_ok=True)

sites = set(get_site_ids(316).keys())
badsites = {'AMT031a', 'DRX008b', 'DRX021a', 'DRX023a', 'ley074a', 'TNC010a'}  # empirically decided
no_perm = {'ley058d'}  # sites without permutations
sites = sites.difference(badsites).difference(no_perm)
print(f'all sites: \n{sites}\n')



recacheDF = True
if fr_DF_file.exists() and not recacheDF:
    DF = jl.load(fr_DF_file)
    ready_sites = set(DF.site.unique())
    sites = sites.difference(ready_sites)
    print('appening new sites to existing DF', sites)
    to_concat = [DF, ]
else:
    to_concat = list()



for site in tqdm(sites):

    trialR, goodcells = load_site_formated_raster(site, contexts='all', probes='all', raster_fs=meta['raster_fs'],
                                                  part='all')


    rep, chn, ctx, prb, tme = trialR.shape

    ctx_nms = range(0, ctx)
    prb_nms = range(1, prb + 1)

    # take the mean across repetitions
    R = np.mean(trialR, axis=(0))  # shape chn x ctx x prb x tme

    # slices either the contexts responses or the probe response
    half_bin = int(trialR.shape[-1]/2) # defines transision between contexte and probe
    part_dict = {'context': np.s_[..., :half_bin], 'probe': np.s_[...,half_bin:]}
    for part_name, part_slicer in part_dict.items():
        part_R = R[part_slicer] # shape chn x ctx x prb x tme

        # averages across reasonable chunks of the response
        q = int(part_R.shape[-1]/4)
        chunk_dict = {'A': np.s_[...,:q], 'B':np.s_[...,q:q*2], 'C':np.s_[...,q*2:q*3], 'D':np.s_[...,q*3:]}
        for chunk_name, chunk_slicer in chunk_dict.items():

            chunk_R = part_R[chunk_slicer].mean(axis=-1) # shape chn x ctx x prb

            df = list()

            # gets the average firing rate for an instance
            for (uu, unit), (cc, ctx), (pp, prb) in itt.product(
                    enumerate(goodcells), enumerate(ctx_nms), enumerate(prb_nms)):
                d = {'id': unit,
                     'site': unit.split('-')[0],
                     'chunk': chunk_name,
                     'part': part_name,
                     'context': ctx,
                     'probe': prb,
                     'metric': 'fr',
                     'value': chunk_R[uu, cc, pp]}
                df.append(d)

            if part_name == 'probe':
                # get the overal probe mean firing rate independent of context
                probe_R = chunk_R.mean(axis=1) # shape chn x pro

                for (uu, unit), (pp, prb) in itt.product(enumerate(goodcells), enumerate(range(1, prb + 1))):
                    d = {'id': unit,
                         'site': unit.split('-')[0],
                         'chunk': chunk_name,
                         'part': part_name,
                         'context': 'mean',
                         'probe': prb,
                         'metric': 'fr',
                         'value': probe_R[uu, pp]}
                    df.append(d)

            to_concat.append(pd.DataFrame(df))


DF = pd.concat(to_concat, ignore_index=True, axis=0)

dups = np.sum(DF.duplicated().values)
if dups > 0:
    print(f'{dups} duplicated rows, what is wrong?, dropping duplicates')
    DF.drop_duplicates(inplace=True)

print(DF.head(10), DF.shape)
jl.dump(DF, fr_DF_file)
