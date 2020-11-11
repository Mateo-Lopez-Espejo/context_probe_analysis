import itertools as itt

import numpy as np
from scipy import stats as sst

from src.data.rasters import _extract_triplets_sub_arr


def pairwise_PSHT_distance(probes, context_transitions, full_array, context_names, probe_names):
    '''
    for each probe, for each cell  Calculates PSTH absolute distance between pairs of contexts.
    Calculatese simple significance based on overlap of SEM.
    Returns an array of pairwise distances plus significance with the shape
    Probe x Context1 x Context2 x Unit x Time x (distance, significance(bool))
    an a
    :param probes: list of probe numbers e.g. [1, 2, 3, 4]
    :param context_transitions: list of triplet transitions names e.g. ['silence', 'continuous' ...]
    :param full_array: nd arrays with shape Context x Probe x Repetition x Unit x Time
    :param context_names: order of the contexts in the full array
    :param probe_names: order of the probes in the full array
    :return: nd array with shape Probe x Context1 x Context2 x Unit x Time x (Distance, Significance(bool))
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
            arr1 = _extract_triplets_sub_arr(probe, ctx1, full_array, context_names, probe_names)  # shape Rep x Unit x Time
            arr2 = _extract_triplets_sub_arr(probe, ctx2, full_array, context_names, probe_names)

            psth1 = np.mean(arr1, axis=0)  # shape Unit x Time
            psth2 = np.mean(arr2, axis=0)
            SEM1 = sst.sem(arr1, axis=0)
            SEM2 = sst.sem(arr2, axis=0)

            distance = np.absolute(psth1 - psth2)
            significance = distance > (SEM1 + SEM2)

            pair_diff_arr[pp, c1, c2, :, :, 0] = distance
            pair_diff_arr[pp, c1, c2, :, :, 1] = significance

    return pair_diff_arr