import collections as col
import numpy as np
from nems import epoch as nep
from nems.recording import Recording
from nems.signal import PointProcess

"""
Collection of functions to identify and separate different stimulation paradigms (triplets or all_permuations) when 
they are loaded together
"""

def split_recording(recording, parameters):
    '''
    split recording into independent recordings for CPP and CPN, does this to all composing signals
    :param recording: a nems.Recording object
    :return:
    '''

    def _split_signal(signal, parameters):
        # finds in epochs the transition between one experiment and the next
        assert type(signal) is PointProcess

        epochs = signal.epochs
        epoch_names = nep.epoch_names_matching(epochs, '\AFILE_[a-zA-Z]{3}\d{3}[a-z]\d{2}_[ap]_CPN\Z')
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

            # check structure and keeps track of how many signal from each type in dict key
            exp_type = parameters[ff]['TrialObject'][1]['ReferenceHandle'][1]['SequenceStructure'].strip()

            if exp_type == 'AllPermutations':
                sig_key = f'perm{perm_counter}'
                perm_counter += 1
            elif exp_type == 'Triplets':
                sig_key = f'trip{trip_counter}'
                trip_counter += 1
            else:
                raise ValueError(f'unknown SequenceStructure {exp_type}')

            sub_signals[sig_key] = sub_signal

        return sub_signals

    rec_dict = col.defaultdict(dict)
    metas = dict()
    for signame, signal in recording.signals.items():
        sub_signals = _split_signal(signal, parameters)

        for sig_type, sub_signal in sub_signals.items():
            rec_dict[sig_type][signame] = sub_signal
            metas[sig_type] = sub_signal.meta
        pass

    rec_dict = {sig_type: Recording(signals, meta=metas[sig_type]) for sig_type, signals in
                      rec_dict.items()}

    return rec_dict
