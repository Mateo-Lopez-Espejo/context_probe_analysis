import numpy as np

from nems import epoch as nep
from nems.signal import SignalBase


def _epoch_name_handler(rec_or_sig, epoch_names):
    '''
    helper function to transform heterogeneous inputs of epochs names (epoch names, list of epochs names, keywords) into
    the corresponding list of epoch names.
    :param rec_or_sig: nems recording of signal object
    :param epoch_names: epoch name (str), regexp, list of epoch names, 'signle', 'pair'. keywords 'single' and 'pair'
    correspond to all signle vocalization, and pair of context probe vocalizations.
    :return: a list with the apropiate epoch names as found in signal.epoch.name
    '''
    if epoch_names == 'single':  # get eps matching 'voc_x' where x is a positive integer
        reg_ex = r'\Avoc_\d'
        epoch_names = nep.epoch_names_matching(rec_or_sig.epochs, (reg_ex))
    elif epoch_names == 'pair':  # get eps matching 'Cx_Py' where x and y are positive integers
        reg_ex = r'\AC\d_P\d'
        epoch_names = nep.epoch_names_matching(rec_or_sig.epochs, (reg_ex))
    elif isinstance(epoch_names, str):  # get eps matching the specified regexp
        reg_ex = epoch_names
        epoch_names = nep.epoch_names_matching(rec_or_sig.epochs, (reg_ex))
    elif isinstance(epoch_names, list):  # uses epoch_names as a list of epoch names.
        ep_intersection = set(epoch_names).intersection(set(rec_or_sig.epochs.name.unique()))
        if len(ep_intersection) == 0:
            raise AttributeError("specified eps are not contained in sig")
        pass

    if len(epoch_names) == 0:
        raise AttributeError("no eps match regex '{}'".format(reg_ex))

    return epoch_names


def _channel_handler(mat_or_sig, channels):
    '''
    Helper function to handle heterogeneous inputs to channel parameter (index, list of indexes or cell names, keywords)
    and returns an homogeneous list of indexes.
    :param mat_or_sig: 3d matrix with shape R x C x T (rep, chan, time), or signal object.
    :param channels: Channel index (int) or list of index, cell name (str) or list of names, 'all'. keyword 'all' includes
    all channels/cells in the signal/matrix.
    :return: list of channels indexes.
    '''
    # chekcs the object type of the parameters
    if isinstance(mat_or_sig, np.ndarray):
        max_chan = mat_or_sig.shape[1]

    elif isinstance(mat_or_sig, SignalBase):
        is_sig = True
        max_chan = mat_or_sig.nchans

    # returns a different list of channles depending on the keywords or channels specified. add keywords here!
    if channels == 'all':
        plot_chans = list(range(max_chan))
    elif isinstance(channels, int):
        if channels >= max_chan:
            raise ValueError('recording only has {} channels, but channels value {} was given'.
                             format(max_chan, channels))
        plot_chans = [channels]
    elif isinstance(channels, list):
        item = channels[0]
        # list of indexes
        if isinstance(item, int):
            for chan in channels:
                if chan > max_chan:
                    raise ValueError('signal only has {} channels, but channels value {} was given'.
                                     format(max_chan, channels))
            plot_chans = channels
        # list of cell names
        elif isinstance(item, str):
            if is_sig != True:
                raise ValueError('can only use cell names when indexing from a signal object')
            plot_chans = [mat_or_sig.chans.index(channels)]

    elif isinstance(channels, str):
        # accepts the name of the unit as found in cellDB
        if is_sig != True:
            raise ValueError('can only use cell names when indexing from a signal object')
        plot_chans = [mat_or_sig.chans.index(channels)]

    return plot_chans