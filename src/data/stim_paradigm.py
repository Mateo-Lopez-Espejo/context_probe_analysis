import collections as col

import numpy as np

from nems import epoch as nep
from nems.recording import Recording
from nems.signal import PointProcess, RasterizedSignal, TiledSignal

"""
Collection of functions to identify and separate different stimulation paradigms (triplets or all_permuations) when 
they are loaded together
"""

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
    structure_epochs = epochs.loc[epochs.name.isin(['AllPermutations', 'Triplets']), :]

    sub_signals = dict()
    trip_counter = 0
    perm_counter = 0

    for ff, (_, structure) in enumerate(structure_epochs.iterrows()):

        # extract relevant epochs and data
        sub_epochs = epochs.loc[(epochs.start >= structure.start) & (epochs.end <= structure.end), :].copy()
        sub_epochs[['start', 'end']] = sub_epochs[['start', 'end']] - structure.start

        sub_data = {cell: spikes[np.logical_and(spikes >= structure.start, spikes < structure.end)] - structure.start
                    for cell, spikes in signal._data.copy().items()}

        sub_signal = signal._modified_copy(data=sub_data, epochs=sub_epochs)

        # keeps track of number of trip of perm experiments
        # names the signal with the experiment type and number in case of repeated trip and/or perm
        if structure['name'] == 'AllPermutations':
            exp_type = f'perm{perm_counter}'
            perm_counter += 1
        elif structure['name'] == 'Triplets':
            exp_type = f'trip{trip_counter}'
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