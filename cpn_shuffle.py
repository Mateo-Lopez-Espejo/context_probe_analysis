import numpy as np


def shuffle_along_axis(array, shuffle_axis, indie_axis=None):
    '''
    shuffles in place an array along the selected axis or group of axis .
    :param array:
    :param shuffle_axis:
    :param indie_axis:
    :return:
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