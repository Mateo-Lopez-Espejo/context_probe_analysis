from configparser import ConfigParser
import pathlib as pl

import joblib as jl
import pandas as pd

from src.data.rasters import load_site_formated_raster
from src.root_path import config_path
from src.utils.subsets import good_sites

from src.models.decoder import unfold_rep_ctx_prb, get_svm_accuracy
from src.data.diagonalization import diag_and_scale
from src.data.region_map import region_map

from publication.globals import RASTER_META


# todo standarize dataframe generating code

# run decoder analysis for all sites and caches on dataframe
config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

# quick cache
acc_df_file = pl.Path(
    config['paths']['analysis_cache']) / f'230220_SC_decoder_accuracies'

recache_acc = False

if acc_df_file.exists() and not recache_acc:
    print('DF cache found, reloading')
    accuracy_df = jl.load(acc_df_file)

elif (not acc_df_file.exists()) or recache_acc:
    print('creating DF of site decoder accuracies ...')
    accuracy_df = list()
    for site in good_sites:
        # for site in [eg_site]:
        fn = load_site_formated_raster
        if fn.check_call_in_cache(site, **RASTER_META):
            print(f'cache found for {site}')
            raster, _ = fn(site, **RASTER_META)
        else:
            print(f"cant load {fn} with {RASTER_META}."
                  f"\n this should be cached, why is it failing? ")

        nsounds = raster.shape[3]
        codes = dict(sparse=raster,
                     dense=diag_and_scale(raster, mode='mean_var'))

        for code, rast in codes.items():
            unfolded, labels = unfold_rep_ctx_prb(rast)
            for part in ['context', 'probe']:
                accuracy, chance = get_svm_accuracy(unfolded, labels[part])

                d = {'site': site,
                     'code': code,
                     'part': part,
                     'accuracy': accuracy,
                     'chance': chance,
                     'nsounds': nsounds}

                accuracy_df.append(d)

    accuracy_df = pd.DataFrame(accuracy_df)
    print('... done creating DF')

    jl.dump(accuracy_df, acc_df_file)

accuracy_df['region'] = accuracy_df['site']
accuracy_df.replace({'region': region_map}, inplace=True)

for col in ['site', 'code', 'part', 'chance', 'nsounds', 'region']:
    accuracy_df[col] = accuracy_df[col].astype('category')


