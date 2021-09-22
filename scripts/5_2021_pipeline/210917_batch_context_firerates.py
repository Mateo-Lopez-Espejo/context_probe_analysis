from src.metrics.consolidated_dprimes import single_cell_dprimes, probewise_dPCA_dprimes, full_dPCA_dprimes
from src.data.load import set_name, load_with_parms
from src.root_path import  config_path
from src.metrics.reliability import signal_reliability
from src.data.rasters import raster_from_sig
from src.data.dPCA import format_raster, _cpp_dPCA


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

rec_recache = False
dprime_recache = False

alpha=0.05

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None,
        'stim_type': 'permutations'}

ctx_fr_DF_file = pl.Path(config['paths']['analysis_cache']) / f'210917_context_firing_rate_DF' / set_name(meta)

analysis_functions = {'SC': single_cell_dprimes, #'LDA':probewise_LDA_dprimes,
                      'pdPCA': probewise_dPCA_dprimes, 'fdPCA': full_dPCA_dprimes}

expt = {'contexts': list(range(11)),
        'probes': list(range(1,11))}

multiple_corrections = {'consecutive_2': ([3], 2),
                        'consecutive_3': ([3], 3),
                        'consecutive_4': ([3], 4)}

metrics = ['significant_abs_mass_center', 'significant_abs_sum']

# sites = set(get_site_ids(316).keys())
sites = {'TNC013a', 'TNC014a', 'TNC015a', 'TNC016a', 'TNC017a', 'TNC018a'}
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a' }  # empirically decided
sites = sites.difference(badsites)


def _load_site_formated_context(site, contexts, probes, meta, part='probe', recache_rec=False):
    """
    wrapper of wrappers. Load a recording, selects the subset of data (triplets, or permutations), generates raster using
    selected  probes and transitions and swaps dimentions into default order
    :param site:
    :contexts:
    :probes:
    :param meta:
    :param recache_rec:
    :return:
    """

    recs, _ = load_with_parms(site, rasterfs=meta['raster_fs'], recache=recache_rec)
    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    # pulls the right recording depending on stimulus type and pulls the signal from it.
    if meta['stim_type'] == 'triplets':
        type_key = 'trip0'
    elif meta['stim_type'] == 'permutations':
        type_key = 'perm0'
    else:
        raise ValueError(f"unknown stim type, use 'triplets' or 'permutations'")

    sig = recs[type_key]['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_sequence*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = raster_from_sig(sig, probes=probes, channels=goodcells, contexts=contexts,
                                              smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                              zscore=meta['zscore'], part=part)

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = format_raster(raster)

    return trialR, R, goodcells

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
    trialR, _, goodcells = _load_site_formated_context(site, **expt, meta=meta)
    DF = DF.append(firing_rates_to_DF(trialR, goodcells, meta),ignore_index=True)

DF.drop_duplicates(inplace=True)

if ctx_fr_DF_file.parent.exists() is False:
    ctx_fr_DF_file.parent.mkdir()
jl.dump(DF, ctx_fr_DF_file)




