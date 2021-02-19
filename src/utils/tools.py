from math import sqrt, log

import numpy as np
from collections import Iterable

import pandas as pd
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

def zscore(array, axis):
    """
    Calculates the z-score of an array using mean and std along all the specified axis. those axis not included in the
    parameter have their z-score calculated independently from one another.
    :param array: ndarray
    :param axis: int or tuple of ints
    :return:
    """
    mean = np.mean(array, axis=axis, keepdims=True)
    std = np.std(array, axis=axis, keepdims=True)
    zscore = (array-mean)/std
    return zscore



def shuffle_along_axis(array, shuffle_axis, indie_axis=None, rng=None):
    '''
    shuffles in place an array along the selected axis or group of axis .
    :param array: nd-array
    :param shuffle_axis: int or int list. axis along which to perform the shuffle
    :param indie_axis: int or int list. shuffling will be done independently across positions in these axis.
    :rng: instance of numpy.random.default_rng(), if none is passed, a random seed is used to create one.
    :return: shuffled array of the same shape as input array.
    '''

    # turn axis inputs into lists of ints.
    if isinstance(shuffle_axis, int):
        shuffle_axis = [shuffle_axis]
    if isinstance(indie_axis, int):
        indie_axis = [indie_axis]

    if rng is None:
        rng = np.random.default_rng()

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
        rng.shuffle(array)
    else:
        # slices the array along the independent axis
        # shuffles independently for each slice
        for ndx in np.ndindex(shape[:len(indie_axis)]): # this is what takes for ever.
            rng.shuffle(array[ndx])

    # reshapes into original dimensions
    array = np.reshape(array, shape)

    # swap the axis back into original positions
    array = np.transpose(array, np.argsort(new_order))

    return array


def ndim_array_to_long_DF(array, label_dict):
    """
    turns an ndimensional array into a long format pandas dataframe, where the axis position on the array are translated
    into labels, and the values of the array are the value column in the dataframe. This is particularly usefull to calculate
    metrics in a vectorized maner instead of using for loops for multiple units or other parameters in a recording site
    dataset
    :param array: ndarray of values with D dimensions
    :param label_dict: dictionary of array labels, with D entries of lists of the corresponding dimension length.
    :return: pandas dataframe with D label columns and one value column.
    """
    flat_labels = np.empty([array.size, array.ndim], dtype=object)
    repeat_num = 1
    for ll, lab in enumerate(label_dict.values()):
        tile_num = int(array.size / (len(lab) * repeat_num))
        flat_lab = np.tile(np.repeat(lab, repeat_num), tile_num)
        repeat_num *= len(lab)
        flat_labels[:, ll] = flat_lab
    flat_array = np.concatenate([flat_labels, array.flatten(order='F')[:, None]], axis=1)

    columns = list(label_dict.keys())
    columns.append('value')

    DF = pd.DataFrame(flat_array, columns=columns)
    return DF