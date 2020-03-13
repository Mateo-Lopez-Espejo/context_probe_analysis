from math import sqrt, log

import numpy as np
from scipy.ndimage import gaussian_filter1d


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


def zscore2(array):
    """
    harcoded zscore for this appliction
    calculates the zscore along the Trial, Stim and Time axis. doing it independently for each cell
    :param array: nd array of shape Rep x Cell x Stimulus(context) x Time
    :return: zscore array of the same shape as input
    """
    means = np.mean(array, axis=(0, 2, 3))[None, :, None, None]
    stds = np.std(array, axis=(0, 2, 3))[None, :, None, None]

    zscore = np.nan_to_num((array - means) / stds)

    return zscore


def zscore(array, indie_axis):
    """
    Calculate the z score of each value in the sample, relative to the sample mean and standard deviation, along the
    specified axis
    :param array: ndarray
    :param indie_axis: int, [int,...]. Axis over which to perform the zscore independently, e.g. Cell axis
    :return: z-scored ndarray
    """

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
    zscore_axis = [x for x in range(array.ndim) if x not in indie_axis]
    new_order = indie_axis + zscore_axis

    array = np.transpose(array, new_order)

    # if multiple axes are being zscored together, reshapes  collapsing across the zscored axis
    # shape of independent chunks of the array, i, o , independent, zscore.
    shape = array.shape
    i_shape = shape[0:len(indie_axis)]
    z_shape = (np.prod(shape[len(indie_axis):], dtype=int),)
    new_shape = i_shape + z_shape

    array = np.reshape(array, new_shape)

    # calcualtes the zscore
    means = np.mean(array, axis=-1)[:, None]
    stds = np.std(array, axis=-1)[:, None]
    zscore = np.nan_to_num((array - means) / stds)

    # reshapes into original dimensions
    zscore = np.reshape(zscore, shape)

    # swap the axis back into original positions
    zscore = np.transpose(zscore, np.argsort(new_order))

    return zscore


def shuffle_along_axis(array, shuffle_axis, indie_axis=None):
    '''
    shuffles in place an array along the selected axis or group of axis .
    :param array: ndarray
    :param shuffle_axis: int or int list. axis along wich to perform the shuffle
    :param indie_axis: int or int list. shuffling will be done independently across positions in these axis.
    :return: shuffled array of the same shape as input array.
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