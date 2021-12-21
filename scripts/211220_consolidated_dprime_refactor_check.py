import itertools as itt
import pathlib as pl
from joblib import Memory
from configparser import ConfigParser

from src.data.rasters import load_site_formated_raster
from src.root_path import config_path

from src.visualization.fancy_plots import quantified_dprime
import src.metrics.consolidated_dprimes as old
import src.metrics.consolidated_dprimes as new
from src.metrics.significance import _significance

"""
this is a done deal and i have confirmed consistency between old and new refactored version.
If you ever wanna check the old versions consolidated (god bless you) checkout git at:
581fd26b22fda6e5e50d11ca298df60dfc415bd7
make sure to turn off the joblib memory.cache for src.data.rasters.load_site_formated_raster

"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
memory = Memory(str(pl.Path(config['paths']['analysis_cache']) / 'consolidated_dprimes'))



# boilerplate paremeters
meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 10,
        'zscore': True,
        'dprime_absolute': None,
        'stim_type': 'permutations'}

expt= {'contexts':list(range(5)),
       'probes':list(range(1,5))}


# good example neuron, original image saved on resports 210414_DAC4.ipynb
cellid= 'ARM021b-36-8'
site = 'ARM021b'
ctx0, ctx1 = 0, 1
probe = 3


def plot_example(dprime, shuff_dprime_quantiles, goodcells):
    significance, confidence_interval = _significance(dprime, shuff_dprime_quantiles,
                                                      multiple_comparisons_axis=[3], consecutive=3, alpha=0.05)

    ctx0, ctx1 = 0, 1

    cell_idx = goodcells.index(cellid) if len(cellid) > 7 else 0
    probe_idx = expt['probes'].index(probe)
    trans_pair_idx = [f'{t0}_{t1}' for t0, t1 in itt.combinations(expt['contexts'], 2)
                      ].index(f'{ctx0}_{ctx1}')
    cell_idx = 0
    fig, ax = quantified_dprime(dprime[cell_idx, trans_pair_idx, probe_idx, :],
                                confidence_interval[:, cell_idx, trans_pair_idx, probe_idx, :],
                                significance[cell_idx, trans_pair_idx, probe_idx, :],
                                raster_fs=meta['raster_fs'])

    fig.show()
    return None


dprime, shuff_dprime_quantiles, goodcells, _ = old.full_dPCA_dprimes(site, contexts='all', probes='all', meta=meta)
plot_example(dprime, shuff_dprime_quantiles, goodcells)

dprime, shuff_dprime_quantiles, goodcells, _ = new.full_dPCA_dprimes(site, contexts='all', probes='all', meta=meta,
                                                                     load_fn=load_site_formated_raster)
plot_example(dprime, shuff_dprime_quantiles, goodcells)

