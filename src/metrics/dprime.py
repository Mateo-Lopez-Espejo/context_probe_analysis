import itertools as itt

import numpy as np


def ndarray_dprime(array0, array1, axis, flip=None, keepdims=False):
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

    # due to floating point error, variances that should be zero are really small numbers, which lead to really big
    # dprimes, this happens most of the time due zero spikes counted
    dprime[dprime > 100000] = 0

    # multiple options to flip the dprime
    if flip == 'absolute':
        dprime = np.abs(dprime)

    elif flip == 'max':
        # flip value signs so the highest absolute dprime value is positive
        toflip = (np.abs(np.min(dprime, axis=-1)) > np.max(dprime, axis=-1))[..., None]  # assume last dimension is time
        dprime = np.negative(dprime, where=toflip, out=dprime)

    elif flip == 'first':
        # flips value signs so the first value in time is positive.
        toflip = (dprime[..., 0] < 0)[..., None]  # assumes last dimension is time
        dprime = np.negative(dprime, where=toflip, out=dprime)

    elif flip is None:
        pass

    return dprime


def pairwise_dprimes(array, observation_axis, condition_axis, flip=None, keepdims=False):
    """
    calculates the dprime in an array where different conditions and different observations correspond to two of the
    dimension of the array.
    :array: ndarray with at least 2 dimensions e.g Montecarlos x Observations x Units x Conditions X Time
    :observation_axis: int. which axis correspond to repeated observations.
    :conditions_axis: int. which axis correspond to the conditions to be paired and compared.
    :return: array of pairwise correlations whith the paired dprimes along the same dimension as the coditions compared
     e.g. (MOntecarlos) x (Repetitions) x Units x Context_Pairs x (Probes) x Time
    """

    dprimes = list()
    for c0, c1 in itt.combinations(range(array.shape[condition_axis]), 2):
        arr0 = np.expand_dims(array.take(c0, axis=condition_axis), axis=condition_axis)
        arr1 = np.expand_dims(array.take(c1, axis=condition_axis), axis=condition_axis)
        dprimes.append(ndarray_dprime(arr0, arr1, axis=observation_axis, flip=flip, keepdims=True))

    # stack the condition pairs along the same dimension of the original conditions
    dprimes = np.stack(dprimes, axis=condition_axis).squeeze(axis=condition_axis+1)

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
    else:
        flipped_montecarlo = montecarlo.copy()
        flipped_montecarlo = np.negative(montecarlo, where=mont_toflip, out=flipped_montecarlo)

    return flipped_dprime, flipped_montecarlo
