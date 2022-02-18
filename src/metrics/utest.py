import itertools as itt
import pathlib as pl
from collections import defaultdict
from configparser import ConfigParser
from math import factorial

import numpy as np
import scipy.stats as sst
from joblib import Memory

from src.data.rasters import load_site_formated_raster
from src.metrics import dprime as cDP
from src.metrics.significance import _signif_quantiles, _raw_pvalue
from src.root_path import config_path
from src.utils.tools import shuffle_along_axis as shuffle

"""
attempt at implementation of the cluster corrected statistical test.
this using the mann whitney U tests as a solid alternative to dprime
which assume normality of the two samples 
"""

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

memory = Memory(str(pl.Path(config['paths']['analysis_cache']) / 'utest'))
print(f'utest functions cache at:\n{memory.location}')

def pairwise_utest(array, observation_axis, condition_axis):

    newshape = list(array.shape)
    newshape[condition_axis] = int(factorial(newshape[condition_axis]) /
                                  (factorial(newshape[condition_axis]-2)* factorial(2)))
    newshape.pop(observation_axis)
    uvalues = np.empty(newshape, dtype=float)

    for cpn, (c0, c1) in enumerate(itt.combinations(range(array.shape[condition_axis]), 2)):
        # this slicing, while more complicated that array.take, but it gives views and not copies thus is more memory efficient
        c0idx =(slice(None),) * condition_axis + (c0,) + (slice(None),) * (array.ndim - condition_axis - 1)
        c1idx =(slice(None),) * condition_axis + (c1,) + (slice(None),) * (array.ndim - condition_axis - 1)

        arr0 = array[c0idx]
        arr1 = array[c1idx]

        uidx = (slice(None),) * (condition_axis-1) + (cpn,) + (slice(None),) * (array.ndim - condition_axis - 1)
        uvalues[uidx], _ = sst.mannwhitneyu(arr0, arr1, alternative='two-sided', axis=observation_axis)

    return uvalues


    # # Prealocates output with shape Neuron x Context-pair x Probe x Time
    # newshape = np.asarray(array.shape)
    # newshape[2] = int(factorial(newshape[2]) / (factorial(newshape[2]-2)* factorial(2)))
    # newshape = newshape[1:]
    # uvalues = np.empty(newshape, dtype=float)
    #
    # for cpn, (c0, c1) in enumerate(itt.combinations(range(array.shape[2]), 2)):
    #     arr0 = array[:,:,c0,:,:]
    #     arr1 = array[:,:,c1,:,:]
    #
    #     uval, pval = sst.mannwhitneyu(arr0, arr1, alternative='two-sided', axis=0)
    #
    #     uvalues[:,cpn,:,:] = uval
    #
    # return uvalues









if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # id = 'TNC010a'  # A1 10 sounds
    # id = 'ARM022a' # PEG 4 sounds
    id = 'ARM021b'  # A1 4 sounds

    raster, goodcells = load_site_formated_raster(id)

    pass
