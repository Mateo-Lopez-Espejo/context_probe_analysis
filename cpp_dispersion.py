import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
from math import factorial
import joblib as jl
import nems.epoch as nep
import scipy.stats as sst
import cpp_parameter_handlers as hand
import itertools as itt

def _single_cell_dispersion(matrixes, channels='all'):
    '''
    given a dictionary of matrixes (from signal.extract_epochs), calculates the variance withing each vocalization pair
    and then between vocalization pairs. these calculations are done over time i.e. for each time bin
    :param matrix:
    :cell:
    :return:
    '''
    # stacks all matrixes (different vocalizations) across new axis, then selects the desired cell
    full_mat = np.stack(matrixes.values(), axis=3)

    # handles channel keywords
    channels = hand._channel_handler(full_mat[..., 0], channels)

    # initializes result matrix with shape C x T where C is cell and T is time
    shape = full_mat.shape
    kruscal_over_time = np.zeros([len(channels), shape[2]]) #empty array of dimentions Cells x Time

    # iterates over cell
    for cell, time in itt.product(channels, range(shape[2])):

        working_slice =  full_mat[:, cell, time, :].T # the resulting array has dimentions Context x Repetition

        try :
            kruscal = sst.kruskal(*working_slice)
            pval = kruscal.pvalue
        except:
            pval = np.nan
        kruscal_over_time[cell, time] = pval

    return kruscal_over_time



def _significance_criterion(dispersion_vector, window, threshold=0.01 ):

    #

    '''
    acording to Asari and sador, to determine significance of a contextual effect, and to avoid false possitive
    due to multiple comparisons, significant differences are only acepted if there are streches of consecutive time bins
    all with significance < 0.01
    :param dispersion_vector:
    :param window:
    :param threshold:
    :return:
    '''

    def rolling_window(a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    windowed = rolling_window(dispersion_vector, window)

    sign_bins = np.where(windowed<=threshold, True, False) # which individual time bins are significant

    sign_wind =  np.all(sign_bins, axis=1) # which windows contain only significant bins

    significant = np.any(sign_wind)

    return significant, sign_wind


def _pairwise_distance_within(matrix):

    # matrix has shape R x C x T: Repetition, Channel/Cell, Time

    # iterates over time

    dispersion = np.zeros([2, matrix.shape[2]]) # empty matrix with shape D x T where D is Dispersion metrics and T is time

    for tt in range(matrix.shape[2]):
        time_slice = matrix[...,tt]
        dist_mat = pairwise_distances(time_slice, metric='euclidean') # input [n_samples, n_features]

        # get the upper triangle of the distance matrix as a flat vector for further calculations
        triU = dist_mat[np.triu_indices(dist_mat.shape[0], 1)]

        # organizes the mean distance and variance in its respective time bin
        dispersion[0, tt] = np.mean(triU)
        dispersion[1, tt] = np.var(triU)

    return dispersion


def _pairwise_distance_between(matrixes):

    # inputs a list of matrixes corresponding to different
    if isinstance(matrixes, dict): #todo figure out how to keep identity
        epochs = matrixes.keys()
        matrixes = matrixes.values()

    # organizes different contexts as a new dimention. The resulting matrix has shape R x N x C X T: Repetition, coNtext,
    # Channel/Cell and Time.
    hyper_matrix = np.stack(matrixes, axis=1)

    # takes the mean across repetitions
    meaned_mat = np.mean(hyper_matrix, axis=0)

    # the resulting matrix has the apropiate shape to be plugged into _pairwise_distance_within i.e. contexts are replacing
    # repetitions
    quick_disp = _pairwise_distance_within(meaned_mat)

    c = meaned_mat.shape[0]
    pair_num = int(((c * c) - c) / 2)
    dispersion = np.zeros([pair_num, meaned_mat.shape[2]])  # empty matrix with shape D x T where D is Dispersion metrics and T is time

    for tt in range(meaned_mat.shape[2]):
        time_slice = meaned_mat[..., tt]
        dist_mat = pairwise_distances(time_slice, metric='euclidean')  # input [n_samples, n_features]

        # get the upper triangle of the distance matrix as a flat vector for further calculations
        triU = dist_mat[np.triu_indices(dist_mat.shape[0], 1)]

        # organizes the mean distance and variance in its respective time bin
        dispersion[:, tt] = triU



    return dispersion

### signal wrapers

def signal_single_cell_dispersion(signal, epoch_names='single', channels='all'):

    # handles epoch_names as standard
    epoch_names = hand._epoch_name_handler(signal, epoch_names)

    # handles channels/cells
    channels = hand._channel_handler(signal, channels)

    matrixes = signal.rasterize().extract_epochs(epoch_names)

    disp = _single_cell_dispersion(matrixes, channels=channels)

    return disp












