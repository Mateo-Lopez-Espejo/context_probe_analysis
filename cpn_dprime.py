import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import pandas as pd
from itertools import permutations
import scipy.fftpack as fp
import scipy.signal as ss
import logging
import itertools as itt



# base functionse
def dprime(array0, array1, absolute=True):
    '''
    calculates the unidimensional timewise dprime between two two-dimensional arrays with shape Trial x Time
    :param array0: ndarray
    :param array1: ndarray
    :absolute: bool. wether to return the absolute d' (direction agnostic) or the signed d'
    :return: 1D array with shape Time
    '''

    if absolute is True:
        dprime = (np.abs(np.mean(array0, axis=0) - np.mean(array1, axis=0)) /
                  np.sqrt(0.5 * (np.var(array0, axis=0) + np.var(array1, axis=0))))
    elif absolute is False:
        dprime = (np.mean(array0, axis=0) - np.mean(array1, axis=0) /
                  np.sqrt(0.5 * (np.var(array0, axis=0) + np.var(array1, axis=0))))
    else:
        raise ValueError(f'absolute must be bool but is {type(absolute)}')

    # check for edge cases
    if np.any(np.logical_or(np.isnan(dprime), np.isinf(dprime))):

        for ii in np.where(np.logical_or(np.isnan(dprime), np.isinf(dprime)))[0]:

            if ((array0[:, ii].mean() - array1[:, ii].mean()) != 0) & \
                    ((np.var(array0[:, ii]) + np.var(array1[:, ii])) == 0):

                # print("Inf. case")
                dprime[ii] = abs(array0[:, ii].mean() - array1[:, ii].mean())

            elif ((array0[:, ii].mean() - array1[:, ii].mean()) == 0) & \
                    ((np.var(array0[:, ii]) + np.var(array1[:, ii])) == 0):

                dprime[ii] = 0

            else:
                raise SystemError('WTF?')

    return dprime


def ndim_dprime(array0, array1, absolute=True):
    '''
    calculates the multidimensional timewise dprime between two tri-dimensional arrays with shape Trial x Cell x Time
    the multidimentiona dprime is defined as the euclidean distance
    :param array0: ndarray
    :param array1: ndarray
    :return: 1D array with shape time
    '''

    # iterates over each neuron and calculates the d'
    all_dprimes = list()
    for cell in range(array0.shape[1]):
        this_dprime = dprime(array0[:, cell, :], array1[:, cell, :], absolute=absolute)
        all_dprimes.append(this_dprime)
    all_dprimes = np.stack(all_dprimes)

    # claculates the ndim hypotenuse fromm each cell d' (per time bin)
    ndim_dprime = np.sqrt(np.sum(np.square(all_dprimes), axis=0))

    return ndim_dprime


def param_sim_resp(array, **kwargs):
    # ToDo complete function
    '''
    Calculates center and dispersion of trial responses from each neuron at each time point,
    simulates new reponses based on these parameters (assumes gaussian distributions).
    :param array: ndarray with shape Trial x Cell x Time
    :param kwargs:
    :return: ndarray with the same shape as the input array. Parametric simulation of neuronal responses.
    '''
    sim_resp = np.random.normal(np.mean(array, axis=0),
                                np.std(array, axis=0),
                                size=array.shape)
    return sim_resp


# full array pairwise functions

def pairwise_dprimes(array):
    '''
    asumes array with shape Repetition x Context x Time, it lacks a Neuron or PC dimention, since the array
    is assumed to come from a projection to a 1d space
    :param array: array of paired dprimes with shape Pairs x Time
    :return: array of paiwise correlations, list of pair id
    '''
    dprimes = list()
    for c0, c1 in itt.combinations(range(array.shape[1]),2):
        dprimes.append(dprime(array[:,c0,:], array[:,c1,:]))

    dprimes = np.stack(dprimes, axis=0)

    return dprimes

# montecarlo functions

def pair_ctx_shuffle_dprime(array, montecarlo):
    return None


def pair_sim_dprimes(array, montecarlo):
    return None
