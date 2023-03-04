import pathlib as pl
from configparser import ConfigParser

from sklearn.decomposition import PCA
import numpy as np
from joblib import Memory

from src.data.rasters import load_site_formated_raster

from src.root_path import config_path

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
PCA_memory = Memory(str(pl.Path(config['paths']['tensors_cache']) / 'PCA'))


@PCA_memory.cache(ignore=['recache_rec'])
def load_site_formated_PCs(site, part='probe', recache_rec=False, **kwargs):
    """
    wrapper of wrappers. Load a recording, selects the subset of data (triplets, or permutations), generates raster using
    selected  probes and transitions
    defaults are the most sensitive after lots of fiddling
    :param site: str e.g. "TNC050"
    :param contexts: list[int] or "all",
    :param probes: list[int] or "all",
    :param meta: dict with fields: "raster_fs": samps/s, "reliability" r^2, "smooth_window":ms ,
    "stim_type": 'triplets' or 'permutations', "zscore": boolean.
    :param part: "context", "probe", "all" default "probe"
    :param recache_rec: boolean, Default False
    :return: raster with shape Repetitions x Pricipal_components x Contexts x Probes x Time_bins,
    dict with PC names and fractional variance explained
    """

    # Load full raster but uses only the probe to fit the PCA.
    _ = load_site_formated_raster.call(site, part='all', recache_rec=recache_rec, **kwargs)
    raster, goodcells = load_site_formated_raster(site, part='all', recache_rec=recache_rec, **kwargs)

    raster_fit = raster[...,int(raster.shape[-1]/2):]

    assert len(raster.shape) == 5
    rep, neu, ctx, prb, tme = raster_fit.shape

    # fits data on trial averages, we don't want to fit noise.
    reshaped = raster_fit.mean(axis=0).reshape([neu, ctx*prb*tme])
    pca = PCA(n_components=0.9)
    pca.fit(reshaped.T)


    # transforms all data, i.e. every trial, not their mean, and all time points
    rep, neu, ctx, prb, tme = raster.shape

    # swap axes and collapese across all dimensions but neuron, does PCA, and reshapes back to normal.
    transformed = np.swapaxes(raster, 0, 1).reshape([neu, -1])
    transformed = pca.transform(transformed.T).T
    nPCs = transformed.shape[0] # automatically detected to capture n_components of variance
    transformed = np.swapaxes(transformed.reshape([nPCs, rep, ctx, prb, tme]), 0,1)

    if part == 'context':
        transformed = transformed[...,:int(raster.shape[-1]/2)]
    elif part == 'probe':
        transformed = transformed[...,int(raster.shape[-1]/2):]
    elif part == 'all':
        pass
    else:
        raise ValueError(f"unrecognized part argument {part}. Use 'context', 'probe' or 'all'")

    # saves PC names with asociated explained variance ratio
    PCs = {f"{site}-PC-{n+1}": var_rate for n, var_rate in enumerate(pca.explained_variance_ratio_)}
    return transformed, PCs


# helper funtions to prepare canonical 5 dim arrays for and from PCA
def unfold_for_PCA(arr):
    rep, neu, ctx, prb, tme = arr.shape
    return np.swapaxes(arr, 0, 1).reshape([neu, -1])

def refold_from_PCA(arr, shape):
    rep, neu, ctx, prb, tme = shape
    # notice how the first two dimension are swapped during reshape
    return np.swapaxes(arr.reshape([neu, rep, ctx, prb, tme]), 0,1)



if __name__ == "__main__":
    cellid, contexts, probe = 'TNC024a-27-2', [1, 10], 3
    PCs, vars = load_site_formated_PCs(site=cellid[:7], part='all')
    raster, goodcells = load_site_formated_raster(site=cellid[:7], part='all')

    import matplotlib.pyplot as plt

    cellid, contexts, probe = 'TNC024a-27-2', [1, 10], 3

    plt.figure()
    plt.plot(PCs[:, 4, contexts[0], probe-1,:].T, color='blue', alpha=0.5)
    plt.plot(PCs[:, 4, contexts[1], probe-1,:].T, color='orange', alpha=0.5)
    plt.show()

    plt.figure()
    plt.plot(raster[:, goodcells.index(cellid), contexts[0], probe - 1, :].T, color='blue', alpha=0.5)
    plt.plot(raster[:, goodcells.index(cellid), contexts[1], probe - 1, :].T, color='orange', alpha=0.5)
    plt.show()

    pass

