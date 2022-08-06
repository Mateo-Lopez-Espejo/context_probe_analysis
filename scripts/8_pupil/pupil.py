import pathlib as pl
from configparser import ConfigParser

import joblib as jl
from tqdm import tqdm
import numpy as np
import pandas as pd

from src.utils.subsets import good_sites
from src.root_path import config_path
from src.data.load import load


config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))


"""

"""
# acumulate all pupil traceses, caches for speed

no_pupil= {'ley070a', 'CRD003b', 'ley075b', 'ley072b', 'CRD002a'}
pupil_traces_file = pl.Path(config['paths']['analysis_cache']) / f'220802_pupil_traces'
pupil_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220802_pupil_summary_DF'

recahce_pup = False

if pupil_traces_file.exists() and not recahce_pup:
    pupil_traces = jl.load(pupil_traces_file)
    sites_to_add = good_sites.difference(set(pupil_traces.keys())).difference(no_pupil)

else:
    pupil_traces = dict()
    sites_to_add = good_sites.difference(no_pupil)


bads = list()
for site in tqdm(sites_to_add):
    # raw pupil trace
    try:
        recs, _ = load(site, rasterfs=20, recache=False, pupil=True)
    except:
        print(f'no pupil trace for {site}')
        bads.append(site)
        continue
    pup_trace = recs['perm0']['pupil']._data.squeeze(axis=0)
    pupil_traces[site] = pup_trace

print(f'cannot find pupil on sites:\n{bads}')
if len(sites_to_add) > 0:
    jl.dump(pupil_traces, pupil_traces_file)

# coefficient of variation and clasification
print('calculating coefficient of variation...')
var_coef = {key:np.std(pup_trace) / np.mean(pup_trace) for key, pup_trace in pupil_traces.items()}


# organize in a dataframe and clasify by quartiles
pup_df = pd.Series(var_coef, name='CV').rename_axis('site').reset_index()
pup_df = pup_df.sort_values(by='CV', ascending=False)

pup_df['pupil_quality'] = pd.qcut(pup_df['CV'], 4, labels=['worst', 'bad', 'good', 'best'])

jl.dump(pup_df, pupil_DF_file)
print('done')