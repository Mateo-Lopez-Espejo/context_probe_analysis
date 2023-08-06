import numpy as np
import pandas as pd
from scikit_posthocs import posthoc_dunn
from scipy import stats as sst


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
    DF['trans_pair'] = DF['trans_pair'].astype('category')
    return DF


def norm_by_mean(DF):
    """
    grand mean across all data, works to normalize metrics across all data, and
    transform units from integral (z-score * ms) into percentage increase in
    firing rate.

    Args:
        DF: Pandas dataframe

    Returns: Pandas dataframe

    """
    normalizer = DF.groupby(by=['metric'], observed=True).agg(
        grand_mean=('value', np.mean)).reset_index()

    DF = pd.merge(DF, normalizer, on=['metric'], validate="m:1")
    DF['norm_val'] = DF['value'] / DF['grand_mean']

    return DF


def simplify_classified_contexts(DF):
    """
    Given a dataframe with a column corresponding to transitions pairs of the
    type "diff_same", "same_silence, etc, creates 3 new columns boolean
    columns: "silence", "same" and "different", identifying if the named
    contexts are presented in the instance (row).

    Args:
        DF: pandas dataframe

    Returns: pandas dataframe

    """
    # even simpler encoding
    # onehot order: diff(1 and 2) same silence
    mapper = {'diff_diff': (1, 0, 0), 'diff_same': (1, 1, 0),
              'diff_silence': (1, 0, 1), 'same_silence': (0, 1, 1)}

    DF['onehot'] = DF['trans_pair'].replace(mapper)

    onehotdf = pd.DataFrame(DF['onehot'].to_list(),
                            columns=['diff', 'same', 'silence'])
    DF.drop(columns='onehot', inplace=True)
    DF = pd.concat([DF, onehotdf], axis=1)

    return DF


def kruskal_with_posthoc(DF, group_col, val_col):
    """
    Little helper function to run kruskal wallis and dunn post hoc test on a
    long format dataframe.
    Args:
        DF: pandas dataframe
        group_col: str. Name of classifier column
        val_col: str. Name of data values column

    Returns:

    """
    # pools data in a list of vectors to run kruskal
    to_stat = list()
    for kk in DF[group_col].unique():
        to_stat.append(DF.loc[DF[group_col] == kk, val_col].values)
    kruskal = sst.kruskal(*to_stat)
    print(kruskal)

    # uses the og dataframe to run the pairwise posthoc test
    dunn = posthoc_dunn(DF, val_col=val_col, group_col=group_col,
                        p_adjust='bonferroni')
    print(f'Dunn post hoc results\n{dunn}')

    return kruskal, dunn


def add_ctx_type_voc(DF):
    '''
    variation of add classified contexts when ferret vocalization is available
    add 2 columns specifying each context type + vocalization
    e.g. diff-voc, or same-nonvoc and a 3rd column merging both contexs and
    handling redundancies e.g. diff-voc_same-nonvoc
    '''
    ctx = np.asarray([row.split('_') for row in DF.context_pair], dtype=int)
    prb = np.asarray(DF.probe, dtype=int)

    # is vocalization??
    vocalizations = ['ferret_fights_Athena-Violet001',
                     'ferret_fights_Jasmine-Violet001']
    voc0 = DF.named_ctx_0.isin(vocalizations).values
    voc1 = DF.named_ctx_1.isin(vocalizations).values
    voc = np.stack([voc0, voc1], axis=1)

    silence = ctx == 0
    same = ctx == prb[:, None]
    different = np.logical_and(~silence, ~same)

    name_arr = np.full_like(ctx, np.nan, dtype=object)

    # silence cannot be a vocalization, omits
    name_arr[np.logical_and(silence, ~voc)] = f'silence'

    for voc_bool, voc_txt in zip([voc, np.logical_not(voc)],
                                 ['-voc', '-nonvoc']):
        name_arr[np.logical_and(same, voc_bool)] = f'same{voc_txt}'
        name_arr[np.logical_and(different, voc_bool)] = f'diff{voc_txt}'

    DF['class_ctx_0'] = name_arr[:, 0]
    DF['class_ctx_1'] = name_arr[:, 1]

    # sort  along col axis to equate diff-same with same-diff
    name_arr = np.sort(name_arr, axis=1)

    DF['class_pair'] = ['_'.join(sorted(x)) for x in name_arr]

    for col_name in ['class_ctx_0', 'class_ctx_0', 'class_pair']:
        DF[col_name] = DF[col_name].astype('category')

    return DF
