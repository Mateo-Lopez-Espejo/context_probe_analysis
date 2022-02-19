import itertools as itt

import numpy as np
from math import factorial


def ndarray_dprime(array0, array1, axis, keepdims=False):
    """
    general function to calculate the d prime between two arrays with the same shape. the d prime is calculated in
    a dimension wise manner but for the specified axis, which is treated as observations/repetitions
    :param array0: ndarray
    :param array1: ndarray
    :param axis: int. observation axis
    :param flip: str, None (default). 'absolute' returns the absolute value of d primes,
                                      'max' flips the values so the max absolute value is positive,
                                      'first' flips the values so the first time time value is positive
    :return: ndarray with one less dimension as the input arrays
    """

    # main dprime calculation
    dprime = ((np.mean(array0, axis=axis, keepdims=keepdims) - np.mean(array1, axis=axis, keepdims=keepdims)) /
              np.sqrt(0.5 * (np.var(array0, axis=axis, keepdims=keepdims) + np.var(array1, axis=axis, keepdims=keepdims))))

    # check for edge cases
    if np.any(np.isnan(dprime)):
        dprime[np.where(np.isnan(dprime))] = 0

    if np.any(np.isinf(dprime)):
        dprime[np.where(np.isinf(dprime))] = (array0.mean(axis=axis, keepdims=keepdims)
                                              - array1.mean(axis=axis, keepdims=keepdims))[np.isinf(dprime)]

    # due to floating point error, variances that should be zero are tiny numbers, which lead to huge
    # dprimes, this happens most of the time due zero spikes counted
    dprime[dprime > 100000] = 0

    return dprime


def pairwise_dprimes(array, observation_axis, condition_axis, keepdims=False):
    """
    calculates the dprime in an array where different conditions and different observations correspond to two of the
    dimension of the array.
    :array: ndarray with at least 2 dimensions e.g Montecarlos x Observations x Units x Conditions X Time
    :observation_axis: int. which axis correspond to repeated observations.
    :conditions_axis: int. which axis correspond to the conditions to be paired and compared.
    :return: array of pairwise correlations whith the paired dprimes along the same dimension as the coditions compared
     e.g. (MOntecarlos) x (Repetitions) x Units x Context_Pairs x (Probes) x Time
    """
    # drop observations and uses the same axis of conditions as pairs of conditionss
    newshape = list(array.shape)
    newshape[observation_axis] = 1
    newshape[condition_axis] = int(factorial(newshape[condition_axis]) /
                                  (factorial(newshape[condition_axis]-2)* factorial(2)))
    dprimes = np.empty(newshape, dtype=float)

    for cpn, (c0, c1) in enumerate(itt.combinations(range(array.shape[condition_axis]), 2)):
        # this slicing, while more complicated that array.take, but it gives views and not copies thus is more memory efficient
        c0idx =(slice(None),) * condition_axis + (c0,) + (slice(None),) * (array.ndim - condition_axis - 1)
        c1idx =(slice(None),) * condition_axis + (c1,) + (slice(None),) * (array.ndim - condition_axis - 1)

        arr0 = np.expand_dims(array[c0idx], axis=condition_axis)
        arr1 = np.expand_dims(array[c1idx], axis=condition_axis)

        didx = (slice(None),) * condition_axis + (cpn,) + (slice(None),) * (array.ndim - condition_axis - 1)
        dprimes[didx]=ndarray_dprime(arr0, arr1, axis=observation_axis, keepdims=True).squeeze(axis=condition_axis)

    if keepdims is False:
        dprimes = dprimes.squeeze(axis=observation_axis)

    return dprimes


def flip_dprimes(dprime_array, montecarlo=None, flip='first'):
    """
    flips the sign of the dprime over time following the rule specified by the keyword argument flip. If a montecarlo
    arrays is provided, its assumed that is related to the dprime_array, and the montecarlo repetitions are flipped in a
    form consistent with the original data from dprime_array i.e. if a dprime time series signs are flipped, the all the
    correspondent montecarlo repetitions are flipped too, regardless of their actual values.
    :param dprime_array: nd arrays of dprime values over time, where the last dimension is time
    :param montecarlo: nd arrays. same shape as dprime_array but with an extra first dimension of montecarlo repetitions
    :param flip: str. 'first' flips dprimes in time so the first time point is positive
    :return: flipped dpriem_array,  flipped montecarlo array
    """
    # defines a boolean mask of what signs are to be flipped in dprime_array
    if flip == 'first':
        # first value is positive
        toflip = (dprime_array[..., 0] < 0)[..., None]  # asumes last dimension is time

    elif flip == 'max':
        # flip value signs so the highest absolute dprime value is positive
        toflip = (np.abs(np.min(dprime_array, axis=-1)) > np.max(dprime_array, axis=-1))[
            ..., None]  # asume last dimensio is time
    elif flip == 'sum':
        # flips so the sum of the time series is positive
        toflip = (np.sum(dprime_array, axis=-1) < 0)[..., None]
    elif flip is None:
        toflip = np.empty(dprime_array.shape)
        toflip[:] = False
    else:
        raise ValueError('flip mode not recognized')

    mont_toflip = toflip[None, ...]  # asumes first dimension of montecarlo are the repetitions

    # using the mask flips the sings
    flipped_dprime = dprime_array.copy()
    flipped_dprime = np.negative(dprime_array, where=toflip, out=flipped_dprime)

    if montecarlo is None:
        flipped_montecarlo = None

    elif isinstance(montecarlo, dict):
        flipped_montecarlo = dict()
        for alpha, mont_array in montecarlo.items():
            flipped_array = mont_array.copy()
            flipped_array = np.negative(mont_array, where=mont_toflip, out=flipped_array)
            flipped_montecarlo[alpha] = flipped_array

    elif isinstance(montecarlo, np.ndarray):
        flipped_montecarlo = montecarlo.copy()
        flipped_montecarlo = np.negative(montecarlo, where=mont_toflip, out=flipped_montecarlo)
    else:
        raise ValueError(f'montecarlo must be a dict of arrays, an array or non but is of {type(montecarlo)}')

    return flipped_dprime, flipped_montecarlo



def cluster_mass():
    # defines threshold
    # find values with abs greater than threshold
    # find clusters
    # calculate metric for clusters
    # relate cluster postions to metric
    return


if __name__ == '__main__':


    pass