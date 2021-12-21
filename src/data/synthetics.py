import numpy as np
from rasters import load_site_formated_raster

def homogenize_cxt_modulation(raster, rep_dim, neu_dim, ctx_dim):
    """
    I believe I tested a more readable version of this trasnformation, but here is the summarized trasnformation.
    The functions is effectively taking the mean response sans contextual modulation, and adding the mean contextual
    modulation alone (no probe response ?)
    :param raster:
    :param rep_dim:
    :param neu_dim:
    :param ctx_dim:
    :return:
    """
    raster_sim = np.mean(raster, axis=ctx_dim, keepdims=True) + \
                 np.mean(raster, axis=(rep_dim, neu_dim), keepdims=True) - \
                 np.mean(raster, axis=(rep_dim, neu_dim, ctx_dim), keepdims=True)
    return raster_sim

def load_homogenized_ctx_raster(site, contexts, probes, meta, part='probe', recache_rec=False):
    """
    wrapper of wrappers. Load a recording, selects the subset of data (triplets, or permutations), generates raster using
    selected  probes and transitions, then homogenizes the contextual modulation across neurons, so it is not sparse but
    rather distributed across all neurons in site.
    :param site: str e.g. "TNC050"
    :param contexts: list[int] or "all",
    :param probes: list[int] or "all",
    :param meta: dict with fields: "raster_fs": samps/s, "reliability" r^2, "smooth_window":ms ,
    "stim_Type": 'triplets' or 'permutations', "zscore": boolean.
    :param part: "context", "probe", "all" default "probe"
    :param recache_rec: boolean, Default False
    :return: raster with shape Repetitions x Cells x Contexts x Probes x Time_bins
    """

    raster, goodcells = load_site_formated_raster(site, contexts, probes, meta, part=part, recache_rec=recache_rec)
    homo_ctx_raster = homogenize_cxt_modulation(raster, 0, 1, 2)

    return homo_ctx_raster






