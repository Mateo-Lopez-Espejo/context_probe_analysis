import collections as col
import warnings

import numpy as np

from nems import epoch as nep
from nems.recording import Recording
from nems.signal import PointProcess, RasterizedSignal, TiledSignal

"""
Collection of functions to identify and separate different stimulation paradigms (triplets or all_permuations) when 
they are loaded together
"""
def _detect_type(epoch):
    '''
    Based on the name of stimuli epochs, defines if the experiment was 'triplets' or all 'permutations'
    :param epoch: pandas DF. NEMS epochs
    :return: str. 'trip' or 'perm'
    '''
    warnings.warn('deprecated stim type recording split. pass a parameter file instead')
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


def _split_signal(signal, parameters=None):
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
    epoch_names = nep.epoch_names_matching(signal.epochs, '\AFILE_[a-zA-Z]{3}\d{3}[a-z]\d{2}_[ap]_CPN\Z')
    if len(epoch_names) == 0:
        raise ValueError('Epochs do not contain files matching CPN experiments.')
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


        try:
            meta = signal.meta.copy()
            meta['rawid'] = [meta['rawid'][ff]]
        except:
            meta = None

        sub_signal = signal._modified_copy(data=sub_data, epochs=sub_epochs, meta=meta)

        # checks names of epochs to define triples or permutation
        # keeps track of number of trip of perm experiments
        # names the signal with the experiment type and number in case of repeated trip and/or perm

        if parameters == None:
            exp_type = _detect_type(sub_epochs)

        else:
            exp_type = parameters[ff]['TrialObject'][1]['ReferenceHandle'][1]['SequenceStructure']
            if exp_type == 'AllPermutations': exp_type = 'perm'
            elif exp_type == 'Triplets': exp_type = 'trip'

        if exp_type == 'perm':
            exp_type = f'{exp_type}{perm_counter}'
            perm_counter += 1
        elif exp_type == 'trip':
            exp_type = f'{exp_type}{trip_counter}'
            trip_counter += 1
        else:
            raise ValueError(f'perm or trip, not {exp_type}')

        sub_signals[exp_type] = sub_signal

    return sub_signals


def split_recording(recording, parameters=None):
    '''
    split recording into independent recordings for CPP and CPN, does this to all composing signals
    :param recording: a nems.Recording object
    :return:
    '''

    sub_recordings = col.defaultdict(dict)
    metas = dict()
    for signame, signal in recording.signals.items():
        sub_signals = _split_signal(signal, parameters=parameters)

        for sig_type, sub_signal in sub_signals.items():
            sub_recordings[sig_type][signame] = sub_signal
            metas[sig_type] = sub_signal.meta

        pass

    sub_recordings = {sig_type: Recording(signals, meta=metas[sig_type]) for sig_type, signals in
                      sub_recordings.items()}


    return sub_recordings