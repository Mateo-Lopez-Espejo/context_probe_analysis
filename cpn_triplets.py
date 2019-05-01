import itertools as itt

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sst

from nems import epoch as nep


def make_full_array(signal, experiment='CPN'):
    # extracts and organizes all the data in a 5 dim array with shape Context x Probe x Repetition x Unit x Time
    signal = signal.rasterize()

    if experiment == 'CPP':
        reg_ex = r'\AC[0-4]_P[0-4]\Z' # contexts 0 to 4, probes 0 to 4

    elif experiment == 'CPN':
        reg_ex = r'\AC(0|([5-9]|10))_P([5-9]|10)\Z' # contexts 0 or 5 to 10, probes 5 to 10, not all contexts exists

    else:
        raise ValueError("experiment must be 'CPP' or 'CPN'")

    epoch_names = nep.epoch_names_matching(signal.epochs, (reg_ex))
    context_names = list(set([cp.split('_')[0] for cp in epoch_names]))
    context_names.sort()
    probe_names = list(set([cp.split('_')[1] for cp in epoch_names]))
    probe_names.sort()

    # gets the dimentions of the full array
    R, U, T = signal.extract_epoch(epoch_names[0]).shape
    C = len(context_names)
    P = len(probe_names)

    full_array = np.empty([C, P, R, U, T])
    full_array[:] = np.nan

    invalid_cp = list()
    valid_cp = list()

    for pp, cc in itt.product(range(len(probe_names)), range(len(context_names))):
        cpp = '{}_{}'.format(context_names[cc], probe_names[pp])
        try:
            full_array[cc, pp, :, :, :] = signal.extract_epoch(cpp)
            valid_cp.append(cpp)
        except:
            invalid_cp.append(cpp)
            # print('{} does not exist, skipping'.format(cpp))

    return full_array, invalid_cp, valid_cp, context_names, probe_names


def extract_sub_arr(probe, context_type, full_array, context_names, probe_names):
    '''
    short function to extract the adecuate slices of the full array given a probe and the specified context transitions
    :param probe: 5 to 10
    :param context_type: str silence, continuous, similar, sharp
    :param full_array: nd array with dimensions Context x Probe x Repetition x Unit x Time
    :context_names: list of context names with order consitent with that of full array (output of make make_full_array)
    :probe_names: list of probe names with order consistent with that of full array (output of make make_full_array)
    :return:
    '''
    '''
    original order of sequences
    array([[ 9, 10,  6,  7,  9],
           [10,  9,  7,  6, 10],
           [ 6,  8,  9,  8, 10],
           [ 7,  5,  6,  5,  7]])
    '''

    if probe in [5, 8]:
        raise ValueError('probe cannot be 5 o 8')

    transitions = {'P5': {'silence': 0,
                          'continuous': None,
                          'similar': [6, 7],
                          'sharp': None},
                   'P6': {'silence': 0,
                          'continuous': 5,
                          'similar': 7,
                          'sharp': 10},
                   'P7': {'silence': 0,
                          'continuous': 6,
                          'similar': 5,
                          'sharp': 9},
                   'P8': {'silence': 0,
                          'continuous': None,
                          'similar': 9,
                          'sharp': 6},
                   'P9': {'silence': 0,
                          'continuous': 8,
                          'similar': 10,
                          'sharp': 7},
                   'P10': {'silence': 0,
                           'continuous': 9,
                           'similar': 8,
                           'sharp': 6}}

    # find array indexes based on probe and context transition names
    p = 'P' + str(probe)
    probe_index = probe_names.index(p)
    c = 'C' + str(transitions[p][context_type])
    context_index = context_names.index(c)

    sliced_array = full_array[context_index, probe_index, :, : , :]

    return sliced_array # array with shape Repetitions Units Time


def calculate_pairwise_distance(probes, context_transitions, full_array, context_names, probe_names):
    '''
    ToDO, documentation, organize in independent file
    :param probes:
    :param context_transitions:
    :param full_array:
    :param context_names:
    :param probe_names:
    :return:
    '''

    P = len(probes)
    CT = len(context_transitions)

    _, _, _, U, T = full_array.shape  # Context x Probe x Repetition x Unit x Time

    # inilializese an array to organzie the output of the difference calculation
    # the array has shape  Probe x ContextTransition x ContextTransition x Units x Time x Metric

    pair_diff_arr = np.empty([P, CT, CT, U, T, 2])
    pair_diff_arr.fill(np.nan)

    # for each probe, calculate pairwise differences.

    for pp, probe in enumerate(probes):
        # interates over pairs of contexts
        for ((c1, ctx1), (c2, ctx2)) in itt.product(enumerate(context_transitions), repeat=2):

            arr1 = extract_sub_arr(probe, ctx1, full_array, context_names, probe_names) # shape Rep x Unit x Time
            arr2 = extract_sub_arr(probe, ctx2, full_array, context_names, probe_names)

            psth1 = np.mean(arr1, axis=0) # shape Unit x Time
            psth2 = np.mean(arr2, axis=0)
            SEM1 = sst.sem(arr1, axis=0)
            SEM2 = sst.sem(arr2, axis=0)

            distance = np.absolute(psth1 - psth2)
            significance = distance > (SEM1 + SEM2)

            pair_diff_arr[pp, c1, c2, :, :, 0] = distance
            pair_diff_arr[pp, c1, c2, :, :, 1] = significance


    return pair_diff_arr