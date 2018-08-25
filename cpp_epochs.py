import pandas as pd
import numpy as np
import nems.epoch as ne
import ast

'''
This is a "temporary" cludge. The context probe pari (CPP) sound objects have both the usual events (epochs) defined 
by the baphy convention "Stim , <whatever> , Reference", wich specify the sequences of individual speech modulated noises. 
The events coresponding to these individual sounds are stored as "SubPreStimSilence, SubStim, SubPostStimSilence" 
and therefore at not being automatically pulled by NEMS into epochs.
So far the substim events have equal duration, and their number, order and identity is stated in the epoch name,
This takes advantages of this facts to generate corresponding subepochs. 
'''


def _set_subepochs(epochs):
    '''

    :param epochs: printine CPP recording epochs
    :return: epochs including subepochs
    '''

    # selects the subset of epochs corresponding to sound sequences
    seq_names = [ep_name for ep_name in epochs.name.unique() if ep_name[0:4] == 'STIM']
    if len(seq_names) == 0:
        raise ValueError("no epochs starting with 'STIM'")

    ff_ep_name = epochs.name.isin(seq_names)
    relevant_eps = epochs.loc[ff_ep_name, :]

    PreStimSilence = epochs.loc[epochs.name == 'PreStimSilence', ['start', 'end']].values
    PreStimSilence = PreStimSilence[0, 1] - PreStimSilence[0, 0]
    PostStimSilence = epochs.loc[epochs.name == 'PostStimSilence', ['start', 'end']].values
    PostStimSilence = PostStimSilence[0, 1] - PostStimSilence[0, 0]

    # organizes the subepochs in an array with shape E x S where E is the number of initial epochs, and S is the number
    # of subepochs

    sub_eps = relevant_eps.name.values
    sub_eps = [ast.literal_eval(ep_name[18:]) for ep_name in sub_eps]
    sub_eps = np.asarray(sub_eps)

    # calculates the start and end of each subepochs based on the start and end of its mother epoch
    original_times = relevant_eps.loc[:, ['start', 'end']].values

    # initializes a matrix with shape E x S x C where E is the number of original epochs, Sis the number of subepochs
    #  per epoch and C is the DF columns to be: start, end, name
    split_times = np.zeros((original_times.shape[0], sub_eps.shape[1], 3))

    # iterates over the epoch dimention
    for ee, epoch in enumerate(original_times):
        step = (epoch[1] - epoch[0] - PreStimSilence - PostStimSilence) / sub_eps.shape[1]
        #
        for ss in range(sub_eps.shape[1]):
            # start of subepoch ss
            split_times[ee, ss, 0] = epoch[0] + PreStimSilence + (step * ss)
            # end of subepochs ss
            split_times[ee, ss, 1] = epoch[0] + PreStimSilence + (step * (ss + 1))
            # name
            split_times[ee, ss, 2] = sub_eps[ee, ss]

    # flatens the matrix by the subepoch dimention into a 2d array and then DF
    newshape = [split_times.shape[0] * split_times.shape[1], split_times.shape[2]]
    flat_times = np.reshape(split_times, newshape)

    sub_epochs = pd.DataFrame(data=flat_times, columns=['start', 'end', 'name'])
    sub_epochs.name.replace({num: 'voc_{:d}'.format(int(num)) for num in sub_epochs.name.unique()}, inplace=True)

    # adds the new epochs to the old ones
    new_epochs = epochs.copy()
    new_epochs = new_epochs.append(sub_epochs)

    # formats by sorting, index and column order
    new_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    new_epochs.reset_index(drop=True, inplace=True)
    new_epochs = new_epochs[['start', 'end', 'name']]

    return new_epochs


def set_signal_subepochs(signal):
    new_epochs = _set_subepochs(signal.epochs)
    new_signal = signal._modified_copy(signal._data, epochs=new_epochs)
    return new_signal


def set_recording_subepochs(recording):
    new_recording = recording.copy()
    for name, signal in recording.signals.items():
        new_signal = set_signal_subepochs(signal)
        new_recording[name] = new_signal
    return new_recording
