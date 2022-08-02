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
Extends the calculation of delta firing rates to make a distinction between big and small pupil

furthermore this calculates the delta firign rate in a bin by bin basis, which is more adecuate than doing it after 
the mean for each chunk has been calculated
"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

raster_meta = {'reliability': 0.1,  # r value
               'smoothing_window': 0,  # ms
               'raster_fs': 20,
               'zscore': True,
               'stim_type': 'permutations'}

fr_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220729_pupil_delta_firerates'  # 20hz
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
                                                  smoothing_window=raster_meta['smoothing_window'])

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
    pupil = np.mean(pupil, axis=-1, keepdims=True)
    threshold = np.median(pupil)
    # notices that in numpy masked arrasy, the values marked as True, are masked OUT,
    # and not considered during array operations
    for pup_size in ['small', 'big']:

        if pup_size == 'small':
            mask = np.broadcast_to(pupil >= threshold, trialR.shape)
        elif pup_size == 'big':
            mask = np.broadcast_to(pupil < threshold, trialR.shape)

        # take the mean across repetitions
        R = np.mean(ma.masked_where(mask, trialR), axis=(0))  # shape chn x ctx x prb x tme

        df = list()
        # calculates pairwise differences.
        for pr_idx, (c0, c1) in enumerate(itt.combinations(ctx_nms, r=2)):

            dFR = R[:,c0,...] - R[:,c1,...] # delta firing rate shape chn x prb x tme

            # slices either the contexts responses or the probe response
            half_bin = int(trialR.shape[-1] / 2)  # defines transision between contexte and probe
            part_dict = {'context': np.s_[..., :half_bin], 'probe': np.s_[..., half_bin:]}
            for part_name, part_slicer in part_dict.items():
                part_dFR = dFR[part_slicer]  # shape chn x prb x tme

                # averages across reasonable chunks of the response
                q = int(part_dFR.shape[-1] / 4)
                chunk_dict = {'A': np.s_[..., :q], 'B': np.s_[..., q:q * 2], 'C': np.s_[..., q * 2:q * 3],
                              'D': np.s_[..., q * 3:],
                              'full': np.s_[...]}
                for chunk_name, chunk_slicer in chunk_dict.items():
                    # average of dFR and abs dfr. The abs version should be equivalent to the integral
                    for metric in ['dfr', 'abs_dfr']:
                        if metric == 'dfr':
                            chunk_dFR = part_dFR[chunk_slicer].mean(axis=-1)  # shape chn x prb
                        elif metric =='abs_dfr':
                            chunk_dFR = np.abs(part_dFR[chunk_slicer]).mean(axis=-1)  # shape chn x prb

                        for (uu, unit), (pp, prb) in itt.product(
                                enumerate(goodcells), enumerate(prb_nms)):
                            d = {'id': unit,
                                 'site': unit.split('-')[0],
                                 'pupil': pup_size,
                                 'chunk': chunk_name,
                                 'part': part_name,
                                 'context_pair': f'{c0:02}_{c1:02}',
                                 'probe': prb,
                                 'metric': metric,
                                 'value': chunk_dFR[uu, pp]}
                            df.append(d)

        to_concat.append(pd.DataFrame(df))

DF = pd.concat(to_concat, ignore_index=True, axis=0)

dups = np.sum(DF.duplicated().values)
if dups > 0:
    print(f'{dups} duplicated rows, what is wrong?, dropping duplicates')
    DF.drop_duplicates(inplace=True)
print(f'failed sites: {bads}')
print(DF.head(10))
print(DF.shape)
jl.dump(DF, fr_DF_file)
