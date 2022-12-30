import pathlib as pl
from configparser import ConfigParser

import joblib as jl
import pandas as pd
from tqdm import tqdm

import nems.db as nd
import nems_lbhb.baphy_io as io

from src.root_path import config_path
from src.data.cell_type import cluster_by_metrics, get_waveform_metrics, get_optotag_DF
from src.utils.subsets import all_cells as goodcells


"""
Simple script to run the waveform analysis on all neurons relevant to the CPN analysis
"""


config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))


##### get waveforms from the selected neurons #####
print('loading all waveforms into dataframe ...')

df_file = pl.Path(config['paths']['analysis_cache']) / '220816_CPN_celltype_DF'
recache = True


badcells = list()

if df_file.exists() and recache == False:
    print(f'DF file found\nloading from {df_file} and appending new neurons')
    celltype_DF = [jl.load(df_file)]
    goodcells = set(goodcells).difference(set(celltype_DF[0].id.tolist()))
else:
    print('calculating waveform parameters...')
    celltype_DF = list()
    badcells = list()


for cellid in tqdm(goodcells):

    try:
        mean_waveform = io.get_mean_spike_waveform(cellid, usespkfile=None)
    except:
        badcells.append(cellid)
        continue

    isolation = nd.get_cell_files(cellid).loc[:,'isolation'].unique()
    isolation = isolation[0]
    if mean_waveform.size == 0:
        print(f'cant get {cellid} waveform')
        badcells.append(cellid)
        continue

    sw, ptr, fwhm, es, bs, trough, wf = get_waveform_metrics(mean_waveform)

    df = pd.DataFrame()
    df['cellid'] = (cellid,)
    df['sw'] = (sw,)
    df['ptr'] = (ptr,)
    df['fwhm'] = (fwhm,)
    df['es'] = (es,)
    df['bs'] = (bs,)
    df['trough'] = (trough,)
    df['waveform_norm'] = (wf.tolist(),)
    df['isolation'] = (isolation,)

    celltype_DF.append(df)

celltype_DF = pd.concat(celltype_DF, ignore_index=True)

celltype_DF = cluster_by_metrics(celltype_DF)

# add phototags when possible
pt = get_optotag_DF()
celltype_DF = pd.merge(celltype_DF, pt, on='cellid', how='left')


# format columns names and types
celltype_DF.rename(columns={'cellid':'id'}, inplace=True)
for col in ['id', 'spike_type']:
    celltype_DF[col] = celltype_DF[col].astype('category')

print(badcells)
jl.dump(celltype_DF, df_file)


print('done')
