import collections as col
import itertools as itt
from math import log, sqrt

import numpy as np
from scipy import stats as sst
from scipy.ndimage import gaussian_filter1d

from nems import epoch as nep
from nems.recording import Recording
from nems.signal import PointProcess, RasterizedSignal, TiledSignal

from cpp_parameter_handlers import _channel_handler


def _detect_type(epoch):
    '''
    Based on the name of stimuli epochs, defines if the experiment was 'triplets' or all 'permutations'
    :param epoch: pandas DF. NEMS epochs
    :return: str. 'trip' or 'perm'
    '''

    # hardwired sequences names, since they are invariant across experiments
    permutations = {'STIM_sequence001: 1 , 3 , 2 , 4 , 4',
                    'STIM_sequence002: 3 , 4 , 1 , 1 , 2',
                    'STIM_sequence003: 4 , 2 , 3 , 3 , 1',
                    'STIM_sequence004: 2 , 2 , 1 , 4 , 3',}

    triplets = {'STIM_sequence001: 5 , 6 , 2 , 3 , 5',
                'STIM_sequence002: 6 , 5 , 3 , 2 , 6',
                'STIM_sequence003: 2 , 4 , 5 , 4 , 6',
                'STIM_sequence004: 3 , 1 , 2 , 1 , 3'}

    names = set(epoch.name.unique())

    if names.issuperset(permutations):
        exp_type = 'perm'
    elif names.issuperset(triplets):
        exp_type = 'trip'
    else:
        raise ValueError('unknown epoch type, not permutations nor triplets')

    return exp_type


def _split_signal(signal):
    # finds in epochs the transition between one experiment and the next
    if isinstance(signal, PointProcess):
        pass
    elif isinstance(signal, RasterizedSignal):
        raise NotImplementedError('signal must be a PointPorcess')
    elif isinstance(signal, TiledSignal):
        raise NotImplementedError('signal must be a PointPorcess')
    else:
        raise ValueError('First argument must be a NEMS signal')

    epochs = signal.epochs
    epoch_names = nep.epoch_names_matching(signal.epochs, '\AFILE_[a-zA-Z]{3}\d{3}[a-z]\d{2}_p_CPN\Z')
    file_epochs = epochs.loc[epochs.name.isin(epoch_names), :]

    sub_signals = dict()
    trip_counter = 0
    perm_counter = 0

    for ff, (_, file) in enumerate(file_epochs.iterrows()):

        # extract relevant epochs and data
        sub_epochs = epochs.loc[(epochs.start >= file.start) & (epochs.end <= file.end), :].copy()
        sub_epochs[['start', 'end']] = sub_epochs[['start', 'end']] - file.start

        sub_data = {cell: spikes[np.logical_and(spikes >= file.start, spikes < file.end)] - file.start
                    for cell, spikes in signal._data.copy().items()}

        meta = signal.meta.copy()
        meta['rawid'] = [meta['rawid'][ff]]

        sub_signal = signal._modified_copy(data=sub_data, epochs=sub_epochs, meta=meta)

        # checks names of epochs to define triples or permutation
        # keeps track of number of trip of perm experiments
        # names the signal with the experiment type and number in case of repeated trip and/or perm
        exp_type = _detect_type(sub_epochs)
        if exp_type == 'perm':
            exp_type = f'{exp_type}{perm_counter}'
            perm_counter += 1
        elif exp_type == 'trip':
            exp_type = f'{exp_type}{trip_counter}'
            trip_counter += 1
        else:
            raise ValueError('not Permutations or Triplets')

        sub_signals[exp_type] = sub_signal

    return sub_signals


def split_recording(recording):
    '''
    split recording into independent recordings for CPP and CPN, does this to all composing signals
    :param recording: a nems.Recording object
    :return:
    '''

    sub_recordings = col.defaultdict(dict)
    metas = dict()
    for signame, signal in recording.signals.items():

        sub_signals = _split_signal(signal)

        for sig_type, sub_signal in sub_signals.items():
            sub_recordings[sig_type][signame] = sub_signal
            metas[sig_type] = sub_signal.meta

        pass

    sub_recordings = {sig_type: Recording(signals, meta=metas[sig_type]) for sig_type, signals in
                      sub_recordings.items()}

    return sub_recordings


def raster_smooth(raster, fs, win_ms, axis):
    '''
    Smooths using a gaussian kernele of the specified window size in ms across one axis, usually time
    :param raster: ndarray. spike raster
    :param fs: fequency of samplig for the spike raster
    :param win_ms: kernel size in ms
    :param axis: axis along with to perform the smoothing. Most likely time
    :return:
    '''
    samples = win_ms * fs / 1000
    sigma = samples / sqrt(8 * log(2))  # this is the magic line to convert from samples to sigma
    smooth = gaussian_filter1d(raster, sigma, axis=axis)

    return smooth


def make_full_array(signal, channels='all', smooth_window=None, raster_fs=None):
    '''
    given a CPP/CPN signal, extract rasters and organizes in a 5D array with axes Context x Probe x Repetition x Unit x Time
    :param signal: nems signal. It should have the context probe epochs defined.
    :param channels: str, int, list, 'all'.  Defines what cells to consider
    :param smooth_window: float. gausian kernel size in ms
    :param raster_fs: int. sampling frequency of the output
    :return: nd array.
    '''

    channels = _channel_handler(signal, channels)

    signal = signal.rasterize(fs=raster_fs)

    reg_ex = r'\AC\d_P\d\Z'

    epoch_names = nep.epoch_names_matching(signal.epochs, (reg_ex))
    context_names = list(set([cp.split('_')[0] for cp in epoch_names]))
    context_names.sort()
    probe_names = list(set([cp.split('_')[1] for cp in epoch_names]))
    probe_names.sort()

    # gets the dimentions of the full array
    R, _, T = signal.extract_epoch(epoch_names[0]).shape
    U = len(channels)
    C = len(context_names)
    P = len(probe_names)

    full_array = np.empty([C, P, R, U, T])
    full_array[:] = np.nan

    invalid_cp = list()
    valid_cp = list()

    for pp, cc in itt.product(range(len(probe_names)), range(len(context_names))):
        cpp = '{}_{}'.format(context_names[cc], probe_names[pp])
        try:
            full_array[cc, pp, :, :, :] = signal.extract_epoch(cpp)[:,channels,:]
            valid_cp.append(cpp)
        except:
            invalid_cp.append(cpp)
            # print('{} does not exist, skipping'.format(cpp))

    # gaussian window smooth
    if smooth_window is not None:
        full_array = raster_smooth(full_array, signal.fs, smooth_window, axis=4)
        pass

    return full_array, invalid_cp, valid_cp, context_names, probe_names


def extract_sub_arr(probes, context_types, full_array, context_names, probe_names, squeeze=True):
    '''
    short function to extract the adequate slices of the full array given a probe and the specified context transitions
    returns a copy not a view.
    :param probes: int. for permutations any of [1,2,3,4]. for triplets any of [2,3,5,6]
    :param context_types: str, [str,]. silence, continuous, similar, sharp. 'all' for default order of the four transitions
    :param full_array: nd array with dimensions Context x Probe x Repetition x Unit x Time
    :context_names: list of context names with order consitent with that of full array (output of make make_full_array)
    :probe_names: list of probe names with order consistent with that of full array (output of make make_full_array)
    :channels:
    :return:
    '''
    '''           
    array([[5, 6, 2, 3, 5],
           [6, 5, 3, 2, 6],
           [2, 4, 5, 4, 6],
           [3, 1, 2, 1, 3]])
           
    '''

    if isinstance(probes, int):
        probes = [probes]

    elif isinstance(probes, (list, tuple, set)):
        if all(isinstance(x, int) for x in probes) and all(x not in (1, 4) for x in probes):
            probes = list(probes)
        else:
            raise ValueError('all values in probe must be int and not 1 or 4')
    else:
        raise ValueError('probe must be an int or a list of ints')

    if context_types == 'all':
        context_types = ['silence', 'continuous', 'similar', 'sharp']
    elif isinstance(context_types, str):
        context_types = [context_types]
    elif isinstance(context_types, (list, tuple, set)):
        context_types = list(context_types)

    for ct in context_types:
        if ct not in ['silence', 'continuous', 'similar', 'sharp']:
            raise ValueError(f'{ct} is not a valid keyword')

    transitions = {'P2': {'silence': 0,
                          'continuous': 1,
                          'similar': 3,
                          'sharp': 6},
                   'P3': {'silence': 0,
                          'continuous': 2,
                          'similar': 1,
                          'sharp': 5},
                   'P5': {'silence': 0,
                          'continuous': 4,
                          'similar': 6,
                          'sharp': 3},
                   'P6': {'silence': 0,
                          'continuous': 5,
                          'similar': 4,
                          'sharp': 2}}


    # create an empty array to populate with the probes slices, with contexts ordered by transition type
    # and not by contexte identity

    C, P, R, U, T = full_array.shape # Context x Probe x Repetition x Unit x Time

    sliced_array = np.empty([len(context_types), len(probes), R, U, T])

    for (pp, probe) in enumerate(probes):
        # find array indexes based on probe and context transition names
        p = 'P' + str(probe)
        probe_index = probe_names.index(p)

        C_names = ['C' + str(transitions[p][ct]) for ct in context_types]
        context_indices = np.asarray([context_names.index(c) for c in C_names])

        sliced_array[:, pp, :, :, :] = full_array[context_indices, probe_index, :, :, :].copy()

        if squeeze == True: sliced_array = np.squeeze(sliced_array)

    return sliced_array  # array with shape Context_transition x Probe x Repetitions x Unit x Time


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
            arr1 = extract_sub_arr(probe, ctx1, full_array, context_names, probe_names)  # shape Rep x Unit x Time
            arr2 = extract_sub_arr(probe, ctx2, full_array, context_names, probe_names)

            psth1 = np.mean(arr1, axis=0)  # shape Unit x Time
            psth2 = np.mean(arr2, axis=0)
            SEM1 = sst.sem(arr1, axis=0)
            SEM2 = sst.sem(arr2, axis=0)

            distance = np.absolute(psth1 - psth2)
            significance = distance > (SEM1 + SEM2)

            pair_diff_arr[pp, c1, c2, :, :, 0] = distance
            pair_diff_arr[pp, c1, c2, :, :, 1] = significance

    return pair_diff_arr
