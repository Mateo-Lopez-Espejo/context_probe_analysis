import itertools as itt
import pathlib as pl
from configparser import ConfigParser

import joblib as jl
import numpy as np
import numpy.ma as ma
import pandas as pd
from tqdm import tqdm

from src.data.rasters import load_site_formated_raster
from src.root_path import config_path
from src.utils.subsets import good_sites as sites
from math import factorial

"""
the mean of differences is equal to the difference of of means (for paired values).  A smaller and straighforward 
dataframe can be made by skipping the delta FR, and just calculating the firing rate.

Also in hindsight, the absolute delta FR is a questionable metric, so we can drop it and save us some time/memory e
"""

# ToDo publication streamline since this is fundamental for the pupil analysis.

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

raster_meta = {'reliability': 0.1,  # r value
               'smoothing_window': 0,  # ms
               'raster_fs': 20,
               'zscore': True,
               'stim_type': 'permutations'}

# fr_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220801_pupil_delta_firerates'  # OG
fr_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220808_pupil_fr_by_instance'  # pupil split calculated by instance
# fr_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220901_raw_fr_by_instance'  # newest version without zscoring firing rates
fr_DF_file.parent.mkdir(parents=True, exist_ok=True)

# sites = ['ARM021b']
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

bads = list()

for site in tqdm(sites):

    trialR, goodcells = load_site_formated_raster(site, contexts='all', probes='all',
                                                  part='all',
                                                  raster_fs=raster_meta['raster_fs'],
                                                  reliability=raster_meta['reliability'],
                                                  smoothing_window=raster_meta['smoothing_window'],
                                                  zscore=raster_meta['zscore'])

    rep, chn, ctx, prb, tme = trialR.shape

    ctx_nms = range(0, ctx)
    prb_nms = range(1, prb + 1)

    # split by puil size:
    try:
        pupil, _ = load_site_formated_raster(site, contexts='all', probes='all',
                                           part='all',
                                           raster_fs=raster_meta['raster_fs'],
                                           reliability=raster_meta['reliability'],
                                           smoothing_window=raster_meta['smoothing_window'],
                                           pupil=True)
    except:
        print(f'{site} has no pupil information')
        bads.append(site)
        continue


    # we wanna find instances i.e. cell*ctx*prb that have on average high or low pupil
    # we therefore make the clasification of pupil size on the full instance repetition pupil size
    # furthermore, the split shoud be done independently per instance
    pupil = np.mean(pupil, axis=-1, keepdims=True)
    thresholds = np.median(pupil, axis=0, keepdims=True) # median per trials in each instance.

    # notices that in numpy masked arrays, the values marked as True, are masked OUT,
    # and not considered during array operations
    for pup_size in ['small', 'big', 'full']:

        if pup_size == 'small':
            mask = np.broadcast_to(pupil >= thresholds, trialR.shape)
        elif pup_size == 'big':
            mask = np.broadcast_to(pupil < thresholds, trialR.shape)
        elif pup_size == 'full':
            mask = np.zeros_like(trialR)

        # take the mean across repetitions
        R = np.mean(ma.masked_where(mask, trialR), axis=(0))  # shape chn x ctx x prb x tme

        df = list()

        # slices either the contexts responses or the probe response
        half_bin = int(trialR.shape[-1] / 2)  # defines transision between contexte and probe
        part_dict = {'context': np.s_[..., :half_bin], 'probe': np.s_[..., half_bin:]}
        for part_name, part_slicer in part_dict.items():
            part_R = R[part_slicer]  # shape chn x prb x tme

            # averages across reasonable chunks of the response
            q = int(part_R.shape[-1] / 4)
            chunk_dict = {'A': np.s_[..., :q], 'B': np.s_[..., q:q * 2], 'C': np.s_[..., q * 2:q * 3],
                          'D': np.s_[..., q * 3:],
                          'full': np.s_[...]}
            for chunk_name, chunk_slicer in chunk_dict.items():

                chunk_R = part_R[chunk_slicer].mean(axis=-1)  # shape chn x ctx x prb

                for (uu, unit), (cc, ctx), (pp, prb) in itt.product(
                        enumerate(goodcells), enumerate(ctx_nms), enumerate(prb_nms)):
                    d = {'id': unit,
                         'site': unit.split('-')[0],
                         'chunk': chunk_name,
                         'part': part_name,
                         'context': ctx,
                         'probe': prb,
                         'metric': 'firing_rate',
                         'pupil': pup_size,
                         'value': chunk_R[uu, cc, pp]}
                    df.append(d)

        to_concat.append(pd.DataFrame(df))

DF = pd.concat(to_concat, ignore_index=True, axis=0)

# enforces memory efficient typing
for col in [c for c in DF.columns if c!='value']:
    DF[col] = DF[col].astype('category')
DF['value'] = pd.to_numeric(DF['value'], downcast='float')


dups = np.sum(DF.duplicated().values)
if dups > 0:
    print(f'{dups} duplicated rows, what is wrong?, dropping duplicates')
    DF.drop_duplicates(inplace=True)
print(f'failed sites: {bads}')
print(DF.head(10))
print(DF.shape)
jl.dump(DF, fr_DF_file)
