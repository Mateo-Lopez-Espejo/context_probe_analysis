import numpy as np
import scipy.stats as sst
from cpp_parameter_handlers import _epoch_name_handler

# # Some test data
# from cpn_load import load
# import cpn_triplets as tp
# rec = load('AMT028b')
# signal = rec['resp'].rasterize()
# epoch_names = r'\ASTIM_Tsequence.*'
# full_array, invalid_cp, valid_cp, all_contexts, all_probes =tp.make_full_array(signal,experiment='CPP')
# raster = full_array[:, 1:, :, :, :]
# rep_dim = 2
# protect_dim = 3
# all_probes.pop(0)

def _base_reliability(raster, rep_dim, protect_dim):
    '''

    :param raster: ndim array
    :param rep_dim: int. dimension corresponding to repetitions
    :protect_dim: int. dimension to keep outside of calculations
    :return: ndarray. Contain perasons R for each position in the protect_dim.
    '''

    # reorders dimensions, first is repetitions, second is protected_dim
    raster = np.moveaxis(raster, [rep_dim, protect_dim], [0, -1])
    R = raster.shape[0]
    P = raster.shape[-1]

    # gets two subsamples across repetitions, and takes the mean across reps
    rep1 = np.nanmean(raster[0:-1:2, ...], axis=0)
    rep2 = np.nanmean(raster[1:R+1:2, ...], axis=0)

    resh1 = np.reshape(rep1,[-1, P])
    resh2 = np.reshape(rep2,[-1, P])

    corcoefs = np.empty(P)
    corcoefs[:] = np.nan
    for pp in range(P):
        r = sst.pearsonr(resh1[:,pp], resh2[:,pp])
        corcoefs[pp] = r[0]

    return corcoefs

def signal_reliability(signal, epoch_names, threshold=0.1):
    '''
    high level wrapper, given a signal and an epochs names, calculates the reliability of the response for each channels
    Reliability is simply the correlation coefficient between two subsets of repetitions.
    for this epoch
    :param signal:
    :param epochs:
    :threshold:
    :return:
    '''
    signal = signal.rasterize()
    epoch_names = _epoch_name_handler(signal, epoch_names)

    # get the stacked rasters. array with shape Epoch x Repetition x Channel x Time
    matrixes = np.stack(list(signal.extract_epochs(epoch_names).values()),axis=0)

    r = _base_reliability(matrixes, rep_dim=1, protect_dim=2)
    goodcells = np.asarray(signal.chans)[r>threshold]

    return r, goodcells





