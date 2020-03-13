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


def ndarray_dprime(array0, array1, axis, absolute=True):
    '''
    general fucntion to calculate the dprime between two arrays with the same shape. the dprime is calculated in
    a dimension wise manner but for the specified axis, which is treated asobservations/repetitions
    :param array0: ndarray
    :param array1: ndarray
    :param axis: int. observation axis
    :param absolute: bool. wether to return the absolute d' (direction agnostic) or the signed d'
    :return: ndarray with one less dimension as the input arrays
    '''

    # main dprime calculation
    if absolute is True:
        dprime = (np.abs(np.mean(array0, axis=axis) - np.mean(array1, axis=axis)) /
                  np.sqrt(0.5 * (np.var(array0, axis=axis) + np.var(array1, axis=axis))))
    elif absolute is False:
        dprime = (np.mean(array0, axis=axis) - np.mean(array1, axis=axis) /
                  np.sqrt(0.5 * (np.var(array0, axis=axis) + np.var(array1, axis=axis))))
    else:
        raise ValueError(f'absolute must be bool but is {type(absolute)}')

    # check for edge cases
    if np.any(np.isnan(dprime)):
        dprime[np.where(np.isnan(dprime))] = 0

    if np.any(np.isinf(dprime)):
        if absolute == True:
            dprime[np.where(np.isinf(dprime))] = (np.abs((array0.mean(axis=axis) - array1.mean(axis=axis))))[np.isinf(dprime)]
        elif absolute == False:
            dprime[np.where(np.isinf(dprime))] = (array0.mean(axis=axis) - array1.mean(axis=axis))[np.isinf(dprime)]
        else:
            raise ValueError(f'absolute must be bool but is {type(absolute)}')


    return dprime

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

def pairwise_dprimes(array, observation_axis, conditiont_axis):
    '''
    calculates the dprime in an array where different conditions and different observations correspond to two of the
    dimension of the array.
    :param array: ndarray with at least 2 dimensions
    :observation_axis: int. which axis correspond to repeated observations.
    :conditions_axis: int. which axis correspond to the conditions to be paired and compared.
    :return: array of pairwise correlations
    '''

    dprimes = list()
    for c0, c1 in itt.combinations(range(array.shape[conditiont_axis]),2):

        arr0 = np.expand_dims(array.take(c0, axis=conditiont_axis),axis=conditiont_axis)
        arr1 = np.expand_dims(array.take(c1, axis=conditiont_axis),axis=conditiont_axis)
        dprimes.append(ndarray_dprime(arr0, arr1, axis=observation_axis))

    # stack the condition pairs along a new first dimension, eliminates the dimension of the original conditions
    dprimes = np.stack(dprimes, axis=0).squeeze()

    return dprimes

# montecarlo functions

def pair_ctx_shuffle_dprime(array, montecarlo):
    return None


def pair_sim_dprimes(array, montecarlo):
    return None
