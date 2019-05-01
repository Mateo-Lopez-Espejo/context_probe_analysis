import pandas as pd
import numpy as np
import re
import itertools as itt
import nems.epoch as ne
import ast
from nems.signal import TiledSignal, RasterizedSignal

'''
This is a "temporary" cludge. The stim_num prb pari (CPP) sound objects have both the usual events (eps) defined 
by the baphy convention "Stim , <whatever> , Reference", wich specify the sequences of individual speech modulated noises. 
The events coresponding to these individual sounds are stored as "SubPreStimSilence, SubStim, SubPostStimSilence" 
and therefore at not being automatically pulled by NEMS into eps.
So far the substim events have equal duration, and their number, order and identity is stated in the epoch name,
This takes advantages of this facts to generate corresponding subepochs. 
'''


# base functions

def _set_subepochs(epochs):
    '''

    :param epochs: original Context Probe Pair (CPP) signal/recording epochs
    :return: epochs including subepochs
    '''

    # selects the subset of eps corresponding to sound sequences
    seq_names = [ep_name for ep_name in epochs.name.unique() if ep_name[0:4] == 'STIM']
    if len(seq_names) == 0:
        raise ValueError("no eps starting with 'STIM'")

    ff_ep_name = epochs.name.isin(seq_names)
    relevant_eps = epochs.loc[ff_ep_name, :]

    PreStimSilence = epochs.loc[epochs.name == 'PreStimSilence', ['start', 'end']].values
    PreStimSilence = PreStimSilence[0, 1] - PreStimSilence[0, 0]
    PostStimSilence = epochs.loc[epochs.name == 'PostStimSilence', ['start', 'end']].values
    PostStimSilence = PostStimSilence[0, 1] - PostStimSilence[0, 0]

    # organizes the subepochs in an array with shape E x S where E is the number of initial eps, and S is the number
    # of subepochs

    sub_eps = relevant_eps.name.values
    sub_eps = [[int(ss) for ss in ep_name[18:].split('  ')] for ep_name in sub_eps]
    sub_eps = np.asarray(sub_eps)

    # calculates the start and end of each subepochs based on the start and end of its mother epoch
    original_times = relevant_eps.loc[:, ['start', 'end']].values

    # initializes a matrix with shape E x S x C where E is the number of original eps, S is the number of subepochs
    # per epoch (adding one to include PostStimSilence) and C is the DF columns to be: start, end, name
    split_times = np.zeros((original_times.shape[0], sub_eps.shape[1] + 1 , 3))


    for ee, epoch in enumerate(original_times):
        # defiens the duration of a single vocalization, i.e. step
        step = (epoch[1] - epoch[0] - PreStimSilence - PostStimSilence) / sub_eps.shape[1]
        for ss in range(sub_eps.shape[1]):
            # start of subepoch ss
            split_times[ee, ss, 0] = epoch[0] + PreStimSilence + (step * ss)
            # end of subepochs ss
            split_times[ee, ss, 1] = epoch[0] + PreStimSilence + (step * (ss + 1))
            # name
            split_times[ee, ss, 2] = sub_eps[ee, ss]

        # ads PostStimSilence, Todo make it better not a hack
        # start, equal to the ende of the last subepoch
        split_times[ee, ss+1, 0] = epoch[0] + PreStimSilence + (step * (ss + 1))
        # end, equal to start plus PostStimSilence time
        split_times[ee, ss+1, 1] = epoch[0] + PreStimSilence + (step * (ss + 1)) + PostStimSilence
        # name, stim_num equal to prb of previous subepoch
        split_times[ee, ss+1, 2] = 0



    # flatens the matrix by the subepoch dimention into a 2d array and then DF
    newshape = [split_times.shape[0] * split_times.shape[1], split_times.shape[2]]
    flat_times = np.reshape(split_times, newshape)

    sub_epochs = pd.DataFrame(data=flat_times, columns=['start', 'end', 'name'])
    sub_epochs.name.replace({num: 'voc_{:d}'.format(int(num)) for num in sub_epochs.name.unique()}, inplace=True)

    # adds the new eps to the old ones
    new_epochs = epochs.copy()
    new_epochs = new_epochs.append(sub_epochs)

    # formats by sorting, index and column order
    new_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    new_epochs.reset_index(drop=True, inplace=True)
    new_epochs = new_epochs.loc[:, ['start', 'end', 'name']]

    return new_epochs


def _set_subepochs_context(epochs):
    '''
    adds a set of epoch names containing both the stim_num and prb identity in the format Cx_Py, where x is the id number
    of the contex, and y that of the prb. by convention, x = 0 corresponds to silence as stim_num. The epoch point only
    to the response to the prb, this only adds the information of what preceded the prb i.e. the stim_num.
    :param epochs: a sig eps DF
    :return: a new sig eps with the aditional CPP eps.
    '''
    new_epochs = _set_subepochs(epochs)

    # finds the names of all signle vocalizations
    single_vocs = {voc for voc in new_epochs.name.unique() if voc[0:3] == 'voc'}

    # selects only eps of single vocalization or prestim silence
    ff_vocs = new_epochs.name.isin(single_vocs)
    ff_silence = new_epochs.name == 'PreStimSilence'
    working_epochs = new_epochs.loc[ff_vocs | ff_silence, :].copy()

    # shortens the names of the eps for later on: vocalizations are positive integers and silence is 0
    replace_dict = {voc: voc[4:] for voc in single_vocs}
    replace_dict['PreStimSilence'] = '0'
    name_arr = working_epochs.name.replace(replace_dict).values

    # makes name consisting of the contex and format of consecutive sounds
    pair_names = ['C{}_P{}'.format(context, probe) for context, probe in zip(name_arr, name_arr[1:])]
    pair_names.insert(0, '')

    # puts back in the filtered eps DF
    working_epochs.loc[:, 'newname'] = pair_names

    # only takes the vocalizations as probes (excludes the silence)
    ff_vocs = working_epochs.name.isin(single_vocs)
    working_epochs = working_epochs.loc[ff_vocs, :]
    working_epochs.loc[:, 'name'] = working_epochs.newname
    working_epochs.drop(columns='newname', inplace=True)

    new_epochs = new_epochs.append(working_epochs)

    # formats by sorting, index and column order
    new_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    new_epochs.reset_index(drop=True, inplace=True)
    new_epochs = new_epochs[['start', 'end', 'name']]

    return new_epochs


def _set_subepoch_pairs(epochs):
    '''

       :param epochs: original Context Probe Pair (CPP) signal/recording epochs
       :return: epochs including subepochs
       '''

    # kludge: some experimetns contain both 'all permutations' and 'tripplets sequences'. renames the tripplets
    # sequence to contain numbers different from those in the all permutation site.
    # since the sequence structure is invariant, here its hardwired

    # sequences of triplets as numbered by matlab, and their transformation
    permutations ={'STIM_sequence001: 1 , 3 , 2 , 4 , 4' : 'STIM_Psequence001: 1 , 3 , 2 , 4 , 4',
                   'STIM_sequence002: 3 , 4 , 1 , 1 , 2' : 'STIM_Psequence002: 3 , 4 , 1 , 1 , 2',
                   'STIM_sequence003: 4 , 2 , 3 , 3 , 1' : 'STIM_Psequence003: 4 , 2 , 3 , 3 , 1',
                   'STIM_sequence004: 2 , 2 , 1 , 4 , 3' : 'STIM_Psequence004: 2 , 2 , 1 , 4 , 3'}


    triplets = {'STIM_sequence001: 5 , 6 , 2 , 3 , 5' : 'STIM_Tsequence001: 9 , 10 , 6 , 7 , 9',
                'STIM_sequence002: 6 , 5 , 3 , 2 , 6' : 'STIM_Tsequence002: 10 , 9 , 7 , 6 , 10',
                'STIM_sequence003: 2 , 4 , 5 , 4 , 6' : 'STIM_Tsequence003: 6 , 8 , 9 , 8 , 10',
                'STIM_sequence004: 3 , 1 , 2 , 1 , 3' : 'STIM_Tsequence004: 7 , 5 , 6 , 5 , 7',}

    epochs = epochs.replace(permutations)
    epochs = epochs.replace(triplets)

    # selects the subset of eps corresponding to sound sequences
    seq_names = [ep_name for ep_name in epochs.name.unique() if ep_name[0:4] == 'STIM']
    if len(seq_names) == 0:
        raise ValueError("no eps starting with 'STIM'")

    ff_ep_name = epochs.name.isin(seq_names)
    relevant_eps = epochs.loc[ff_ep_name, :]

    # finds the duration of the prestim and poststim silences
    PreStimSilence = epochs.loc[epochs.name == 'PreStimSilence', ['start', 'end']].values
    PreStimSilence = PreStimSilence[0, 1] - PreStimSilence[0, 0]
    PostStimSilence = epochs.loc[epochs.name == 'PostStimSilence', ['start', 'end']].values
    PostStimSilence = PostStimSilence[0, 1] - PostStimSilence[0, 0]

    # organizes the subepochs in an array with shape E x S where E is the number of initial eps, and S is the number
    # of subepochs

    sub_epochs = relevant_eps.name.values
    # indexes in the 'list' part of the name and split into a list of integers by double space '  '
    sub_epochs = [[int(ss) for ss in ep_name[19:].split(',')] for ep_name in sub_epochs]
    sub_epochs = np.asarray(sub_epochs)

    # calculates the start and end of each subepochs based on the start and end of its mother epoch
    original_times = relevant_eps.loc[:, ['start', 'end']].values

    # initializes a matrix with shape SP x DF where SP is the number of subepochs including both singles and pairs
    # and DF is the DF columns to be: start and end

    total_subepochs = (sub_epochs.size * 2) + sub_epochs.shape[0]  # first terms includes bot signle and pair vocs
    # second term is for PostStimSilence as prb in pairs
    splited_times = np.zeros([total_subepochs, 2])
    new_names = np.empty([total_subepochs, 1], dtype='object')

    # determines the duration of an individual vocalization
    step = (original_times[0, 1] - original_times[0, 0] - PreStimSilence - PostStimSilence) / sub_epochs.shape[1]

    cc = 0
    # iterates over the original epochs
    for ee, (epoch, this_ep_sub_eps) in enumerate(zip(original_times, sub_epochs)):

        # iterates over single subepochs
        for ss, sub_ep in enumerate(this_ep_sub_eps):

            # first add as a single
            # start time
            start = epoch[0] + PreStimSilence + (step * ss)
            splited_times[cc, 0] = start
            # end time
            end = epoch[0] + PreStimSilence + (step * (ss + 1))
            splited_times[cc, 1] = end
            # name
            new_names[cc, 0] = 'voc_{}'.format(sub_ep)

            # second add as a pair
            cc += 1
            # stim_num start time
            if ss == 0:  # special case for PreStimSilence as stim_num
                context = start - PreStimSilence
                name = 'C0_P{}'.format(sub_ep)
            else:
                context = start - step
                name = 'C{}_P{}'.format(this_ep_sub_eps[ss - 1], sub_ep)

            splited_times[cc, 0] = context
            splited_times[cc, 1] = end
            new_names[cc, 0] = name
            cc += 1

        # finally add the PostStimSilences as prb in a pair

        context = start
        end = end + PostStimSilence
        name = 'C{}_P0'.format(sub_ep)

        splited_times[cc, 0] = context
        splited_times[cc, 1] = end
        new_names[cc, 0] = name

        cc += 1

    # Concatenate data array and names array and organizes in an epoch dataframe
    new_data = np.concatenate([splited_times, new_names], axis=1)
    sub_epochs = pd.DataFrame(data=new_data, columns=['start', 'end', 'name'])

    # adds the new eps to the old ones
    new_epochs = epochs.copy()
    new_epochs = new_epochs.append(sub_epochs)

    # formats by sorting, index and column order
    new_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    new_epochs.reset_index(drop=True, inplace=True)
    new_epochs = new_epochs.loc[:, ['start', 'end', 'name']]

    return new_epochs


# sig and recording wrappers

def set_signal_subepochs(signal, set_pairs=True):
    if set_pairs == False:
        new_epochs = _set_subepochs_context(signal.epochs)
    elif set_pairs == True:
        new_epochs = _set_subepoch_pairs(signal.epochs)
    else:
        raise ValueError("keyword argument 'set_pairs' must be a boolean")

    # special case tiled signal copy...
    if isinstance(signal, TiledSignal):
        attributes = signal._get_attributes()
        attributes.update({'epochs':new_epochs})
        new_signal = TiledSignal(data=signal._data, safety_checks=False, **attributes)
    else:
        new_signal = signal._modified_copy(signal._data, epochs=new_epochs)
    return new_signal


def set_recording_subepochs(recording, set_pairs=True):
    new_recording = recording.copy()
    for name, signal in recording.signals.items():
        new_signal = set_signal_subepochs(signal, set_pairs=set_pairs)
        new_recording[name] = new_signal
    return new_recording

def rename_into_part(epochs, context_or_probe='probe'):
    '''
    replace the composite cpp epoch names of the form Cn_Pn, where n is the number identifiying a particular vocalization,
    with eithe the stim_num part Cn or the prb part Cn
    :param epochs: NEMS epochs data frame preformated with CPP epochs
    :param context_or_probe: str 'stim_num' or 'prb', defines wich part of the name to keep
    :return: epochs DF with the renamed epochs
    '''

    cpp_epochs = ne.epoch_names_matching(epochs, r'\AC\d_P\d')
    if not cpp_epochs:
        raise ValueError('epochs do not contain stim_num prb formated epochs')

    if context_or_probe == 'context':
        rename_dict = {old_epoch: old_epoch.split('_')[0] for old_epoch in cpp_epochs}
    elif context_or_probe == 'probe':
        rename_dict = {old_epoch: old_epoch.split('_')[1] for old_epoch in cpp_epochs}

    new_epochs = epochs.replace(rename_dict)

    return new_epochs

