import numpy as np
import itertools as itt
import warnings as warn
from collections import defaultdict as ddict

from cpp_dispersion import _pairwise_single_trial_ndim_euclidean
from cpn_triplets import make_full_array, extract_sub_arr
from cpp_parameter_handlers import _channel_handler

transitions_map = {'P2': {'silence': 'C0',
                          'continuous': 'C1',
                          'similar': 'C3',
                          'sharp': 'C6'},
                   'P3': {'silence': 'C0',
                          'continuous': 'C2',
                          'similar': 'C1',
                          'sharp': 'C5'},
                   'P5': {'silence': 'C0',
                          'continuous': 'C4',
                          'similar': 'C6',
                          'sharp': 'C3'},
                   'P6': {'silence': 'C0',
                          'continuous': 'C5',
                          'similar': 'C4',
                          'sharp': 'C2'}}




inverse_tran_map = {outKey:{inVal:inKey for inKey, inVal in outVal.items()}
                       for outKey, outVal in transitions_map.items()}


def _probe_sanitize(probes):

    valid_probes = {'P2', 'P3', 'P5', 'P6'}
    # sanitizes probe inputs
    if isinstance(probes, str):
        probes = [probes]
    elif isinstance(probes, int):
        probes = ['P'+str(probes)]
    elif isinstance(probes, list):
        temp_probes = list()
        for pp in probes:
            if isinstance(pp, int):
                if 'P'+str(pp) in valid_probes:
                    temp_probes.append('P'+str(pp))
                else:
                    raise ValueError('Probes has invalid numbers')
            elif isinstance(pp, str) and pp in valid_probes:
                temp_probes.append(pp)
            else:
                raise ValueError('values in probes must be valid str or int')
        probes = temp_probes
    else:
        raise ValueError('probe must be an int, str, or a list of them')

    return probes


def _sanitize_transitions(transitions):
    # sanitizes context transition inputs
    valid_transitions = {'silence', 'continuous', 'similar', 'sharp'}
    if isinstance(transitions, str) and transitions in valid_transitions:
        transitions = [transitions]
    elif isinstance(transitions, list):
        for cc in transitions:
            if cc in valid_transitions:
                continue
            else:
                raise ValueError('elements in context_transisions are not valid strings')
    elif transitions is None:
        pass
    else:
        raise ValueError('context_transitions must be a valid keyword or list of them')

    return transitions


def signal_single_trial_dispersion_pooled_shuffled(signal, probe_names, context_transitions=None, channels='all',
                                                   shuffle_num=100, trial_combinations=False):
    # todo documentation

    # sanitizes input parameters
    probe_names = _probe_sanitize(probe_names)
    context_transitions = _sanitize_transitions(context_transitions)
    signal = signal.rasterize()

    # extracts and organizes all the data in a 5 dim array with shape Context x Probe x Repetition x Unit x Time
    full_array, invalid_cp, valid_cp, all_contexts, all_probes = make_full_array(signal, channels=channels)
    T = full_array.shape[-1] # time bins

    # generates indexing masks to call proper pairs of CPPs, probe by probe for distance calculation.
    # distinguish between slicers comparing the same CPPs or different CPPs
    same_slicers = list()
    diff_slicers = list()
    for pname in probe_names:
        # get this probe valid context names
        valid_contexts = [cp.split('_')[0] for cp in valid_cp if cp.split('_')[-1] == pname]
        # based on the context_transitions, boils down to the contexts specific for this probe
        # right now it is working with pairs of transition types
        valid_contexts = [vc for vc in valid_contexts if inverse_tran_map[pname][vc] in context_transitions]
        # put in numeric order to allow skipping repeated combinations
        valid_contexts.sort(key=lambda a:int(a[1:]))

        for cname0, cname1 in itt.product(valid_contexts, repeat=2):
            # avoids calculating the same distance twice dist(a,b) == dist(b,a)
            if int(cname0[1:]) > int(cname1[1:]):
                continue

            # using the contexts and probe names, find the right indices in the array
            cind0 = all_contexts.index(cname0)
            cind1 = all_contexts.index(cname1)
            pind = all_probes.index(pname)

            slicer = np.s_[np.array([cind0, cind1]), pind, :, :]

            if cname0 == cname1:
                same_slicers.append(slicer)
            else:
                diff_slicers.append(slicer)


    # calculates mean distance betwee CPP pairs for eache pair, by their independent groups
    same_ctx_dist = np.empty([len(same_slicers), T])
    for ii, sc in enumerate(same_slicers):
        same_ctx_dist[ii, :] = np.nanmean(_pairwise_single_trial_ndim_euclidean(full_array[sc][0], full_array[sc][1],
                                                                                matrices_equal=True,
                                                                                trial_combinations=trial_combinations),
                                          axis=0)

    diff_ctx_dist = np.empty([len(diff_slicers), T])
    for ii, dc in enumerate(diff_slicers):
        diff_ctx_dist[ii, :] = np.nanmean(_pairwise_single_trial_ndim_euclidean(full_array[dc][0], full_array[dc][1],
                                                                                matrices_equal=False,
                                                                                trial_combinations=trial_combinations),
                                          axis=0)

    # takes the mean across pairs of context-probes for each pool, and takes the difference between pools
    # end result is a 1 dim array of shape TimeBins
    real_diff_min_same = np.nanmean(diff_ctx_dist, axis=0) - np.nanmean(same_ctx_dist, axis=0)

    # shuffles n times across the context dimention to generate the bootstrap distribution
    # copy the array to keep the original unchanged. Todo is this necesary??
    shuffled_diff_min_same = np.empty([shuffle_num, real_diff_min_same.shape[0]])

    shuffle_arr = full_array.copy()

    for jj in range(shuffle_num):
        shuffle_arr = full_array.copy() # C x P x R x U x T
        shuffle_arr = np.swapaxes(shuffle_arr,1,2) # C x R x P x U x T
        s=shuffle_arr.shape
        shuffle_arr = np.reshape(shuffle_arr,(s[0]*s[1],s[2],s[3],s[4])) # (C x R) x P x U x T
        np.random.shuffle(shuffle_arr) # (C x R) x P x U x T
        shuffle_arr = np.reshape(shuffle_arr, s)
        shuffle_arr = np.swapaxes(shuffle_arr,1,2)

        # calculates mean distance between CPP pairs for eache pair, by their independent groups
        shuff_same_ctx_dist = np.empty([len(same_slicers), T])
        for ii, sc in enumerate(same_slicers):
            shuff_same_ctx_dist[ii, :] = np.nanmean(
                _pairwise_single_trial_ndim_euclidean(shuffle_arr[sc][0], shuffle_arr[sc][1],
                                                      matrices_equal=True, trial_combinations=trial_combinations),
                axis=0)

        shuff_diff_ctx_dist = np.empty([len(diff_slicers), T])
        for ii, dc in enumerate(diff_slicers):
            shuff_diff_ctx_dist[ii, :] = np.nanmean(
                _pairwise_single_trial_ndim_euclidean(shuffle_arr[dc][0], shuffle_arr[dc][1],
                                                      matrices_equal=False, trial_combinations=trial_combinations),
                axis=0)

        # takes the mean across pairs of context-probes, end result is a 1 dim array of shape TimeBins
        # calculates the difference
        shuffled_diff_min_same[jj, :] = np.nanmean(shuff_diff_ctx_dist, axis=0) - \
                                        np.nanmean(shuff_same_ctx_dist, axis=0)

    return real_diff_min_same, shuffled_diff_min_same


def shuffle_along_axis(array, shuffle_axis, indie_axis=None):
    '''
    shuffles in place an array along the selected axis or group of axis .
    :param array:
    :param shuffle_axis:
    :param indie_axis:
    :return:
    '''
    # sanitize the shuffle_axis value, it can either be an integer or a list of integers
    if isinstance(shuffle_axis, int):
        shuffle_axis = [shuffle_axis]
    elif isinstance(shuffle_axis, (list, tuple, set)):
        if all(isinstance(x, int) for x in shuffle_axis):
            shuffle_axis = list(shuffle_axis)
        else:
            raise ValueError('all values in shuffle_axis must be int')
    else:
        raise ValueError('shuffle_axis must be an int or a list of ints')

    # sanitize the indie_axis valu, it can be either an integer or a list of integers
    if isinstance(indie_axis, int):
        indie_axis = [indie_axis]
    elif isinstance(indie_axis, (list, tuple, set)):
        if all(isinstance(x, int) for x in indie_axis):
            indie_axis = list(indie_axis)
        else:
            raise ValueError('all values in indie_axis must be int')
    elif indie_axis is None:
        indie_axis = []
    else:
        raise ValueError('indie_axis must be an int or a list of ints')


    # reorder axis, first: indie_axis second: shuffle_axis, third: all other axis i.e. protected axis.
    other_axis = [x for x in range(array.ndim) if x not in indie_axis and x not in shuffle_axis]
    new_order = indie_axis + shuffle_axis + other_axis

    array = np.transpose(array, new_order)

    # if multiple axes are being shuffled together, reshapes  collapsing across the shuffle_axis
    # shape of independent chunks of the array, i, s, o , independent, shuffle, other.
    shape = array.shape
    i_shape = shape[0:len(indie_axis)]
    s_shape = (np.prod(shape[len(indie_axis):len(shuffle_axis)+len(indie_axis)], dtype=int),)
    o_shape = shape[-len(other_axis):] if len(other_axis) > 0 else ()

    new_shape =  i_shape + s_shape + o_shape

    array = np.reshape(array, new_shape)

    if indie_axis is None:
        np.random.shuffle(array)
    else:
        # slices the array along the independent axis
        # shuffles independently for each slice
        for ndx in np.ndindex(shape[:len(indie_axis)]):
            np.random.shuffle(array[ndx])

    # reshapes into original dimensions
    array = np.reshape(array, shape)

    # swap the axis back into original positions
    array = np.transpose(array, np.argsort(new_order))

    return array


array = np.arange(60)
array = np.reshape(array, [3,4,5])

array = np.arange(120)
array = np.reshape(array, [2,3,4,5])

shuffle_along_axis(array, [2,3], 1)
