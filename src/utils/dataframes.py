import numpy as np
import pandas as pd

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


def add_classified_contexts(DF):
    ctx = np.asarray([row.split('_') for row in DF.context_pair], dtype=int)
    prb = np.asarray(DF.probe, dtype=int)

    silence = ctx == 0
    same = ctx == prb[:,None]
    different = np.logical_and(~silence, ~same)

    name_arr = np.full_like(ctx, np.nan, dtype=object)
    name_arr[silence] = 'silence'
    name_arr[same] = 'same'
    name_arr[different] = 'diff'
    name_arr = np.sort(name_arr, axis=1)

    comp_name_arr = np.apply_along_axis('_'.join, 1, name_arr)

    DF['trans_pair'] = comp_name_arr
    return DF