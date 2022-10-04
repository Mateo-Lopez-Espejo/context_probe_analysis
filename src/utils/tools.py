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

def zscore_spont(signal):
    # todo!!
    sile_name = signal.epochs.get_epochs_matching()
    silece_R = signal.extract_epochs(sile_name)
    return None



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


def decimate_xy(x, y, end_num, by_quantiles=True, rng=None):
    """
    Decimates a set of x, y points by a random subset or by the means of n quantiles
    :x: np.vector
    :y: np.vector same shape as x
    :end_num: size of the output vectors
    :by_quantiles: Bool (def. True). use quantiles instead of random subset.
    """
    assert len(x) == len(y)

    if end_num >= len(x):
        print('decimation end number greater than data, returning unchaged data')
        return x, y

    if by_quantiles:
        srtidx = np.argsort(x)
        x = x[srtidx]
        y = y[srtidx]
        qntils = np.quantile(x, np.linspace(0, 1, end_num + 1), interpolation='higher')
        xm = np.empty((end_num))
        ym = np.empty((end_num))
        for rr in range(len(qntils) - 1):
            if rr == 0:
                mask = (x <= qntils[rr + 1])
            else:
                mask = (qntils[rr] < x) & (x <= qntils[rr + 1])

            if np.sum(mask) == 0:
                # empty quantil, place nan
                xm[rr] = np.nan
                ym[rr] = np.nan
            else:
                xm[rr] = np.mean(x[mask])
                ym[rr] = np.mean(y[mask])

        # remove nan quantiles
        xm = xm[~ np.isnan(xm)]
        ym = ym[~ np.isnan(ym)]


    else:
        if rng is None:
            rng = np.random.default_rng()

        decimator = np.random.choice(x.size, end_num, replace=False)
        xm, ym = x[decimator], y[decimator]

    return xm, ym