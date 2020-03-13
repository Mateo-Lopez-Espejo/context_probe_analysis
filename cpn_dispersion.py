import numpy as np
import itertools as itt

from tools import shuffle_along_axis
from cpp_dispersion import _pairwise_single_trial_ndim_euclidean
from cpn_triplets import make_full_array

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

    # shuffles n times across the context dimension to generate the bootstrap distribution
    shuffled_diff_min_same = np.empty([shuffle_num, real_diff_min_same.shape[0]])
    shuffle_arr = full_array.copy()

    for jj in range(shuffle_num):

        # shuffles the context identity and trial
        shuffle_arr = shuffle_along_axis(shuffle_arr, shuffle_axis=[0, 1], indie_axis=None)

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


    # shuffles the trial identity independently across cells i.e. scrambles population response into an uncoordinanted
    # state, to test for relevance of population code.
    population_scramble = np.empty([shuffle_num, real_diff_min_same.shape[0]])
    shuffle_arr = full_array.copy()

    for jj in range(shuffle_num):

        # shuffles trials independently across cells
        # Context x Probe x Repetition x Unit x Time

        shuffle_arr = shuffle_along_axis(shuffle_arr, shuffle_axis=2, indie_axis=3)

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
        population_scramble[jj, :] = np.nanmean(shuff_diff_ctx_dist, axis=0) - \
                                        np.nanmean(shuff_same_ctx_dist, axis=0)

    return real_diff_min_same, shuffled_diff_min_same, population_scramble




def transition_pair_comparison_by_trials(transitions_array, probe_names, context_transitions, probe_order, trans_order,
                                         shuffle_num=100, trial_combinations=False):
    # todo documentation

    C, P, R, U, T = transitions_array.shape

    # truncantes the full transition array to exclude contexts_transitions and probes not called for the analysis, this
    # for the sake of proper shuffling across these axis, which othewise would lead into including undesired data
    trans_slice = np.asarray([trans_order.index(c) for c in context_transitions])
    probe_slice = np.asarray([probe_order.index(p) for p in probe_names])
    transitions_array = transitions_array[trans_slice, probe_slice, :, :, :]
    if len(probe_names) == 1: # restores lost dimension
        transitions_array = np.expand_dims(transitions_array,1)

    # modified probe and trans order consisten with the new sliced array
    trans_order, probe_order = context_transitions, probe_names

    # generates indexing masks to call proper pairs of CPPs, probe by probe for distance calculation.
    # distinguish between slicers comparing the same CPPs or different CPPs
    same_slicers = list()
    diff_slicers = list()
    for probe in probe_names:
        p_index = probe_order.index(probe)
        for cname0, cname1 in itt.product(context_transitions, repeat=2):

            cind0, cind1 = trans_order.index(cname0), trans_order.index(cname1)

            # avoids calculating the same distance twice dist(a,b) == dist(b,a)
            if cind0 > cind1:
                continue

            slicer = np.s_[np.array([cind0, cind1]), p_index, :, :]

            if cname0 == cname1:
                same_slicers.append(slicer)
            else:
                diff_slicers.append(slicer)


    # calculates mean distance betwee CPP pairs for eache pair, by their independent groups
    same_ctx_dist = np.empty([len(same_slicers), T])
    for ii, sc in enumerate(same_slicers):
        same_ctx_dist[ii, :] = np.nanmean(_pairwise_single_trial_ndim_euclidean(transitions_array[sc][0], transitions_array[sc][1],
                                                                                matrices_equal=True,
                                                                                trial_combinations=trial_combinations),
                                          axis=0)

    diff_ctx_dist = np.empty([len(diff_slicers), T])
    for ii, dc in enumerate(diff_slicers):
        diff_ctx_dist[ii, :] = np.nanmean(_pairwise_single_trial_ndim_euclidean(transitions_array[dc][0], transitions_array[dc][1],
                                                                                matrices_equal=False,
                                                                                trial_combinations=trial_combinations),
                                          axis=0)

    # takes the mean across pairs of context-probes for each pool, and takes the difference between pools
    # end result is a 1 dim array of shape TimeBins
    real_diff_min_same = np.nanmean(diff_ctx_dist, axis=0) - np.nanmean(same_ctx_dist, axis=0)

    # shuffles n times across the context dimension to generate the bootstrap distribution
    shuffled_diff_min_same = np.empty([shuffle_num, real_diff_min_same.shape[0]])
    shuffle_arr = transitions_array.copy()

    for jj in range(shuffle_num):

        # shuffles the context identity independently for each trial
        shuffle_arr = shuffle_along_axis(shuffle_arr, shuffle_axis=0, indie_axis=2)

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


    # shuffles the trial identity independently across cells i.e. scrambles population response into an uncoordinanted
    # state, to test for relevance of population code.
    population_scramble = np.empty([shuffle_num, real_diff_min_same.shape[0]])
    shuffle_arr = transitions_array.copy()

    for jj in range(shuffle_num):

        # shuffles trials independently across cells
        # Context x Probe x Repetition x Unit x Time

        shuffle_arr = shuffle_along_axis(shuffle_arr, shuffle_axis=2, indie_axis=3)

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
        population_scramble[jj, :] = np.nanmean(shuff_diff_ctx_dist, axis=0) - \
                                        np.nanmean(shuff_same_ctx_dist, axis=0)

    return real_diff_min_same, shuffled_diff_min_same, population_scramble