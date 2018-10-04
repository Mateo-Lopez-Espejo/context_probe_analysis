import collections as coll
import itertools as itt
import math

import numpy as np
import scipy.stats as sst
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

import cpp_cache as cch
import cpp_parameter_handlers as hand
import cpp_plots as cplt


### helper funtions

def _into_windows(array, window, axis=-1, rolling=True, padding=np.nan, ):
    '''
    I am so proud of this fucntion. Takes an nd array, of N dimensions and generates an array of N+1 of windows across
    the selected dimention. The selected dimention becomes window number, and the new last dimension becomes the original
    dimention units across the window lenght.
    :param array: nd array
    :param window: window size in bins
    :param axis: axis along which to generate the windows
    :param rolling: wheter to use rollinge overlaping windows or static yuxtaposed windows
    :param padding: element used to pad the last windows
    :return: nd.array of n+1 dimensions,
    '''
    if window == 1:
        windowed = np.expand_dims(array, axis=array.ndim)
        return windowed
    elif window == 0:
        raise ValueError('window must be possitive non zero integer')

    # swapps interest axis to last positions for easier indexing
    swapped = np.swapaxes(array, axis1=axis, axis2=-1)
    old_shape = swapped.shape

    # defines paddign
    pad_shape = list(old_shape)

    if rolling == True:
        pad_shape[-1] = pad_shape[-1] + window - 1
    elif rolling == False:
        extra_len = int((window * np.ceil(pad_shape[-1] / window)) - pad_shape[-1])
        pad_shape[-1] = pad_shape[-1] + extra_len

    else:
        raise ValueError('rolling must be a bool')

    padded = np.empty(pad_shape)
    padded[:] = padding
    padded[..., :swapped.shape[-1]] = swapped

    if rolling == True:
        newshape = list(old_shape)
        newshape.append(window)  # just add a new dimention of window length and the end
        windowed = np.empty(newshape)

        for ii in range(old_shape[-1]):
            windowed[..., ii, :] = padded[..., ii:ii + window]

    elif rolling == False:
        old_ax_len = int(pad_shape[-1] / window)  # windowed dimention divided by window len
        newshape = pad_shape
        newshape[-1] = old_ax_len  # modifies old dimention
        newshape.append(window)  # add new dimention i.e. elements along window.
        windowed = padded.reshape(newshape, order='C')  # C order, changes las index the fastest.

    # returs to the "original" dimention order, where the windowed dimention becomes the window dimention i.e. window number
    # and the last dimension becomes the values corresponding to the selected dimnesion, along each window.
    windowed = windowed.swapaxes(axis, -2)  # swaps the original last dimension(-2) back in place

    return windowed


def _collapse_dim_into_dim(argin):
    argout = None
    return argout


def _split_dim_into_dims(argin):
    argout = None
    return argout


### single-cell window dispersion functions

def _window_kruskal(working_window):
    '''
    calculates kruskal wallis between contexts , binning by time,
    for an array of shape Repetition x Context x Time
    :param working_window: 3d ndarray with dims Repetition x Context x Time
    :return: float pvalue
    '''
    # flattens the window time bin dimension into the Repetition dimension
    working_window = np.swapaxes(working_window, 0, 1)
    work_shape = working_window.shape
    working_window = working_window.reshape(
        work_shape[0],
        work_shape[1] * work_shape[2], order='C')  # dimensions = Context x (Repetitions * window Time bin)

    # kruskal Wallis pvalue calculates simultaneously for all contexts
    try:
        kruscal = sst.kruskal(*working_window)
        pval = kruscal.pvalue
    except:
        pval = np.nan

    return pval


def _window_pearsons(working_window):
    '''
    calculates mean of the pairwise pearsons correlation of the PSTHs (i.e. collapsed repetitions) between contexts
    for an array of shape Repetition x Context x Time
    :param working_window: 3d ndarray with dims Repetition x Context x Time
    :return: float pvalue
    '''

    def _working_window_rval(working_window):
        # input array should have shape Repetitions x Context x Time
        # calculates PSTH i.e. mean across repetitions
        psth = working_window.mean(axis=0)  # dimentions Context x WindowTime
        # initializes array for cc of all combinations
        combs = int(math.factorial(psth.shape[0]) / (2 * math.factorial(psth.shape[0] - 2)))
        r_values = np.empty(combs)
        # iterates over every pair of contexts
        for ctx, (ctx1, ctx2) in enumerate(itt.combinations(range(psth.shape[0]), 2)):
            # calculates an holds the correlation coeficient
            r_val, pval = sst.pearsonr(psth[ctx1, :], psth[ctx2, :])
            r_values[ctx] = r_val

        mean_r = r_values.mean()
        return mean_r

    # 1. calculates the mean of the pairwise correlation coefficient between al differente contexts
    obs_rval = _working_window_rval(working_window)

    # 2. shuffles across repetitions a contexts
    # collapses repetition and context together
    collapsed = working_window.swapaxes(0, 2)  # maktes time the first axis, the only relevant to hold
    t, c, r = collapsed.shape
    collapsed = collapsed.reshape([t, c * r], order='C')
    # makes the dimention to suffle first, to acomodate for numpy way of shuffling
    shuffled = collapsed.T  # dimentions (C*R) x T

    shuffle_n = 100

    rval_floor = np.empty([100])

    # n times shuffle

    for rep in range(shuffle_n):
        # shuffles array
        np.random.shuffle(shuffled)
        # reshapes
        reshaped = shuffled.T.reshape(t, c, r).swapaxes(0, 2)
        # calculates pairwise r_value
        rval_floor[rep] = _working_window_rval(reshaped)

    # two sided pvalue
    # pvalue = (rval_floor > obs_rval).sum() / shuffle_n

    if obs_rval > rval_floor.mean():
        pvalue = (rval_floor > obs_rval).sum() / shuffle_n
    elif obs_rval < rval_floor.mean():
        pvalue = (rval_floor < obs_rval).sum() / shuffle_n
    else:
        pvalue = 1

    return pvalue


def _window_MSD(working_window):
    '''
        calculates mean of the pairwise pearsons correlation of the PSTHs (i.e. collapsed repetitions) between contexts
        for an array of shape Repetition x Context x Time
        :param working_window: 3d ndarray with dims Repetition x Context x Time
        :return: float pvalue
        '''

    def _working_window_MSD(working_window):
        # input array should have shape Repetitions x Context x Time
        # calculates PSTH i.e. mean across repetitions
        psth = working_window.mean(axis=0)  # dimentions Context x WindowTime
        # initializes array to hold the calculated metric for all context pairs combinations
        combs = int(math.factorial(psth.shape[0]) / (2 * math.factorial(psth.shape[0] - 2)))
        msd_values = np.empty(combs)
        # iterates over every pair of contexts
        for ctx, (ctx1, ctx2) in enumerate(itt.combinations(range(psth.shape[0]), 2)):
            # calculates an holds the mean standard difference
            msd = np.nanmean((psth[ctx1, :] - psth[ctx2, :]) ** 2)  # uses nanmean to acount for the nan
            # padding at end of rolling window
            msd_values[ctx] = msd

        mean_r = msd_values.mean()
        return mean_r

    # 1. calculates the mean of the pairwise correlation coefficient between all differente contexts
    obs_msd = _working_window_MSD(working_window)

    # 2. shuffles across repetitions a contexts
    # collapses repetition and context together
    collapsed = working_window.swapaxes(0, 2)  # makes time the first axis, the only relevant to hold
    t, c, r = collapsed.shape
    collapsed = collapsed.reshape([t, c * r], order='C')
    # makes the dimention to suffle first, to acomodate for numpy way of shuffling
    shuffled = collapsed.T  # dimentions (C*R) x T

    shuffle_n = 100

    msd_floor = np.empty([100])

    # n times shuffle

    for rep in range(shuffle_n):
        # shuffles array
        np.random.shuffle(shuffled)
        # reshapes
        reshaped = shuffled.T.reshape(t, c, r).swapaxes(0, 2)
        # calculates pairwise r_value
        msd_floor[rep] = _working_window_MSD(reshaped)

    # pvalue = (msd_floor > obs_msd).sum() / shuffle_n

    if obs_msd > msd_floor.mean():
        pvalue = (msd_floor > obs_msd).sum() / shuffle_n
    elif obs_msd < msd_floor.mean():
        pvalue = (msd_floor < obs_msd).sum() / shuffle_n
    else:
        pvalue = 1

    return pvalue


### cell-population window dispersion functions


def _window_ndim_MSD(working_window):
    '''
        calculates mean of the pairwise Mean Standard Difference of the PSTHs (i.e. collapsed repetitions) between contexts
        for an array of shape Repetition x Unit x Context x WindowTimeBin
        :param working_window: 4D array has dimensions Repetition x Unit x Context x WindowTimeBin
        :return: float pvalue
        '''

    def _working_window_MSD(working_window):
        # input array should have shape Repetition x Unit x Context x WindowTimeBin
        # calculates PSTH i.e. mean across repetitions
        psth = working_window.mean(axis=0)  # dimentions Unit x Context x WindowTime
        # initializes array to hold the calculated metric for all context pairs combinations
        combs = int(math.factorial(psth.shape[1]) / (2 * math.factorial(psth.shape[1] - 2)))
        msd_values = np.empty(combs)
        # iterates over every pair of contexts
        for ctx, (ctx1, ctx2) in enumerate(itt.combinations(range(psth.shape[1]), 2)):
            # compares the psth difference for each Unit
            unit_sqr_dif = (psth[:, ctx1, :] - psth[:, ctx2, :]) ** 2
            # sums across Units, means across time
            msd_values[ctx] = np.nanmean(np.sum(unit_sqr_dif, axis=0), axis=0)

        mean_r = msd_values.mean()
        return mean_r

    # 1. calculates the mean of the pairwise correlation coefficient between all differente contexts
    obs_msd = _working_window_MSD(working_window)

    # 2. shuffles across repetitions a contexts
    # collapses repetition and context together
    # first two axes remain unaltered, swaps axes to shape WindowTimeBin x Unit x Context x repetition
    collapsed = working_window.swapaxes(0, 3)
    t, u, c, r = collapsed.shape
    collapsed = collapsed.reshape([t, u, c * r], order='C')
    # makes the dimention to suffle first, to acomodate for numpy way of shuffling
    shuffled = collapsed.swapaxes(0, 2)  # dimentions (C*R) x U x T

    shuffle_n = 100

    msd_floor = np.empty([shuffle_n])

    # n times shuffle

    for rep in range(shuffle_n):
        # shuffles array
        np.random.shuffle(shuffled)
        # reshapes
        reshaped = shuffled.transpose(2, 1, 0).reshape(t, u, c, r).transpose(3, 1, 2, 0)
        # calculates pairwise r_value
        msd_floor[rep] = _working_window_MSD(reshaped)

    # pvalue = (msd_floor > obs_msd).sum() / shuffle_n

    if obs_msd > msd_floor.mean():
        pvalue = (msd_floor > obs_msd).sum() / shuffle_n
    elif obs_msd < msd_floor.mean():
        pvalue = (msd_floor < obs_msd).sum() / shuffle_n
    else:
        pvalue = 1

    shuffle_test = coll.namedtuple('shuffled_vals', 'pvalue calculated floor')

    return shuffle_test(pvalue, obs_msd, msd_floor)


### base single cell dispersion fucntions

def _single_cell_difsig(matrices, channels='all', window=1, rolling=False, type='Kruskal'):
    '''
    given a dictionary of matrices (from signal.extract_epochs), calculates pvalue for a difference metric
    for the response to the different contexts of a stimuli (different epochs, each of the keywords in the dictionary).
    these calculations are done over time i.e. for each time bin, and independently for each cell

    :param matrices: a dictionary of matrices of dimensions Repetitions x Cells x Time. Each keywords corresponds to a different stimulus
    :param channels: the channels/cells to consider (second dimension of input matrices)
    :param window: window size in time bins over which to perform the calculations.
    :param rolling: boolena, wheather to use rolling windows or non overlapping juxtaposed windows
    :param type: keyword defining what metric to use. 'Kruskal' for Kruskal Wallis,
    'Pearsons' for mean of pairwise correlation coefficient.
    :return: an array of shape Cell x Time, of pvalues for each cell across time.
    '''
    # stacks all matrices (different vocalizations) across new axis, then selects the desired cell
    full_mat = np.stack(matrices.values(), axis=3)  # shape: Repetitions x Channels x TimeBins x ContextStimuli

    # handles channel keywords
    channels = hand._channel_handler(full_mat[..., 0], channels)

    # generates the windowed array
    windowed = _into_windows(full_mat, window=window, axis=2, rolling=rolling)

    # initializes result matrix with shape C x T where C is cell and T is time or TimeWindow
    shape = windowed.shape
    metric_over_time = np.zeros([len(channels), shape[2]])  # empty array of dimentions Cells x Time

    # iterates over cell
    for cc, cell in enumerate(channels):
        # iterates over time window
        for wind in range(shape[2]):
            working_window = windowed[:, cell, wind, :,
                             :]  # array has dimensions Repetition x Context x window Time bin

            # selects betwee different metrics
            if type == 'Kruskal':
                pvalue = _window_kruskal(working_window)

            elif type == 'Pearsons':
                if window == 1: raise ValueError('Pearsons correlation requieres window of size > 1')
                pvalue = _window_pearsons(working_window)

            elif type == 'MSD':
                pvalue = _window_MSD(working_window)

            else:
                raise ValueError('keyword {} not suported'.format(type))

            metric_over_time[cc, wind] = pvalue

    return metric_over_time


def _significance_criterion(pvalues, axis, window=1, threshold=0.01, comp='<='):
    '''
    acording to Asari and sador, to determine significance of a contextual effect, and to avoid false possitive
    due to multiple comparisons, significant differences are only acepted if there are streches of consecutive time bins
    all with significance < 0.01. takes an array of pvalues and returns a boolean vector of significance
    acording to an alpha threshold an a window size
    :param pvalues: 2d array of dimentions C x T where C are cells/channels and T are time bins
    :param window: rolling window sizes in number of time bins, default 1 i.e time window = bin size
    :param threshold: certainty threshold, by default 0.01
    :return: boolean array of the same dimentions of pvalues array
    '''

    # windowed = _into_rolling_windows(pvalues, window, padding=np.nan)
    windowed = _into_windows(pvalues, window=window, axis=axis, rolling=True, padding=np.nan)

    # which individual time bins are significant
    if comp == '<=':
        sign_bin = np.where(windowed <= threshold, True, False)
    elif comp == '>=':
        sign_bin = np.where(windowed >= threshold, True, False)
    else:
        raise ValueError(" only '<=' and '>=' defined")

    sign_window = np.all(sign_bin, axis=-1)  # which windows contain only significant bins

    return sign_window


### base population dispersion functions

def _population_difsig(matrices, channels='all', window=1, rolling=False, type='MSD'):
    '''
    given a dictionary of matrices (from signal.extract_epochs), calculates pvalue for a difference metric
    for the response to the different contexts of a stimuli (different epochs, each of the keywords in the dictionary).
    these calculations are done over time i.e. for each time bin and considering all cells together

    :param matrices: a dictionary of matrices of dimensions Repetitions x Cells x Time. Each keywords corresponds to a
    stimulus with a different context.
    :param channels: the channels/cells to consider (second dimension of input matrices)
    :param window: window size in time bins over which to perform the calculations.
    :param rolling: boolena, wheather to use rolling windows or non overlapping juxtaposed windows
    :param type: keyword defining what metric to use. 'Kruskal' for Kruskal Wallis,
    'Pearsons' for mean of pairwise correlation coefficient.
    :return: an array of shape Time, of pvalues for the whole population across time.
    '''
    # stacks all matrices (different vocalizations) across new axis, then selects the desired cell
    full_mat = np.stack(matrices.values(), axis=3)  # shape: Repetitions x Channels x TimeBins x ContextStimuli

    # handles channel keywords
    channels = hand._channel_handler(full_mat[..., 0], channels)

    # generates the windowed array
    # shape REpetition x Channels x Window x ContextStimuli x WindowTimeBin.
    windowed = _into_windows(full_mat, window=window, axis=2, rolling=rolling)

    # initializes result matrix with shape T where T is time or TimeWindow
    shape = windowed.shape
    metric_over_time = np.zeros(shape[2])

    # iterates over time window
    for wind in range(shape[2]):
        working_window = windowed[:, :, wind, :, :]  # array has dimensions Repetition x Cell x Context x WindowTimeBin

        if type == 'MSD':
            pvalue = _window_ndim_MSD(working_window).pvalue

        elif type == 'Euclidean':
            # ToDo implement
            raise NotImplementedError

        else:
            raise ValueError('keyword {} not suported'.format(type))

        metric_over_time[wind] = pvalue

    return metric_over_time


def _pairwise_distance_within(matrix):
    # matrix has shape R x C x T: Repetition, Channel/Cell, Time

    # iterates over time

    dispersion = np.zeros(
        [2, matrix.shape[2]])  # empty matrix with shape D x T where D is Dispersion metrics and T is time

    for tt in range(matrix.shape[2]):
        time_slice = matrix[..., tt]
        dist_mat = pairwise_distances(time_slice, metric='euclidean')  # input [n_samples, n_features]

        # get the upper triangle of the distance matrix as a flat vector for further calculations
        triU = dist_mat[np.triu_indices(dist_mat.shape[0], 1)]

        # organizes the mean distance and variance in its respective time bin
        dispersion[0, tt] = np.mean(triU)
        dispersion[1, tt] = np.var(triU)

    return dispersion


def _pairwise_distance_between(matrixes):
    # inputs a list of matrixes corresponding to different
    if isinstance(matrixes, dict):  # todo figure out how to keep identity
        epochs = matrixes.keys()
        matrixes = matrixes.values()

    # organizes different contexts as a new dimention. The resulting matrix has shape R x N x C X T: Repetition, coNtext,
    # Channel/Cell and Time.
    hyper_matrix = np.stack(matrixes, axis=1)

    # takes the mean across repetitions
    meaned_mat = np.mean(hyper_matrix, axis=0)

    # the resulting matrix has the apropiate shape to be plugged into _pairwise_distance_within i.e. contexts are replacing
    # repetitions
    quick_disp = _pairwise_distance_within(meaned_mat)

    c = meaned_mat.shape[0]
    pair_num = int(((c * c) - c) / 2)
    dispersion = np.zeros(
        [pair_num, meaned_mat.shape[2]])  # empty matrix with shape D x T where D is Dispersion metrics and T is time

    for tt in range(meaned_mat.shape[2]):
        time_slice = meaned_mat[..., tt]
        dist_mat = pairwise_distances(time_slice, metric='euclidean')  # input [n_samples, n_features]

        # get the upper triangle of the distance matrix as a flat vector for further calculations
        triU = dist_mat[np.triu_indices(dist_mat.shape[0], 1)]

        # organizes the mean distance and variance in its respective time bin
        dispersion[:, tt] = triU

    return dispersion


###  signal wrapers

def signal_single_context_sigdif(signal, epoch_names='single', channels='all', dimensions='cell', fs=None, window=1, rolling=False, type='Kruskal', recache=False, signal_name=None):
    func_args = locals()

    # looks in cache
    folder = '/home/mateo/mycache/cpp/'
    name_args = {key: val for key, val in func_args.items() if key not in ['signal', 'signal_name']}
    # acnowledgese its a population calculation
    name_args['population'] = False
    cache_name = cch.set_name(name_args, signal_name=signal_name, onlysig=False)
    cache_out = cch.cache_wrap(obj_name=cache_name, folder=folder, obj=None,
                               recache=recache)
    # if cache exists, returns, else runs the rest of the funciton
    if cache_out is not None:
        return cache_out

    # handles epoch_names as standard
    epoch_names = hand._epoch_name_handler(signal, epoch_names)

    # handles channels/cells
    channels = hand._channel_handler(signal, channels)

    # handles the frequency sampling for analysis
    fs = hand._fs_handler(signal, fs)

    # defines what what frequency sampling to use for the analysis
    if fs == signal.fs:
        tempsig = signal
    else:
        tempsig = signal._modified_copy(signal._data, fs=fs)

    matrixes = tempsig.rasterize().extract_epochs(epoch_names)

    if dimensions == 'population':
        diff_pvals = _population_difsig(matrixes, channels=channels, window=window, rolling=rolling, type=type)

    elif dimensions == 'cell':
        diff_pvals = _single_cell_difsig(matrixes, channels=channels, window=window, rolling=rolling, type=type)

    else:
        raise ValueError("dimensions can only be 'cell' or 'population', but {} was given".format(dimensions))


    # saves to cache
    diff_pvals = cch.cache_wrap(obj_name=cache_name, folder=folder, obj=diff_pvals, recache=recache)

    return diff_pvals


def signal_all_context_sigdif(signal, channels, probes=(1, 2, 3, 4), dimensions='cell', sign_fs=None, window=1, rolling=True,
                              type='Kruskal'):
    '''
    calculates the difference pvalue across all cells in all probes
    :param signal: cpp Singal object
    :param channels: channel index, cell name (or a list of the two previous). or kwd 'all' to define which channels to consider
    :param probes:
    :param sign_fs:
    :param window:
    :param rolling:
    :param type:
    :return:
    '''

    all_probes = list()
    compound_names = list()

    # handlese sign_fs
    sign_fs = hand._fs_handler(signal, sign_fs)

    # calculates dipersion pval for each set of contexts probe.
    for pp in probes:
        this_probe = r'\AC\d_P{}'.format(pp)
        disp_mat = signal_single_context_sigdif(signal, epoch_names=this_probe, channels=channels, dimensions=dimensions,
                                                fs=sign_fs, window=window, rolling=rolling, type=type)

        if dimensions == 'cell':
            chan_idx = hand._channel_handler(signal, channels)  # heterogeneous "channels" value to indexes
            cell_names = [name for nn, name in enumerate(signal.chans) if nn in chan_idx]
            comp_names = ['C*_P{}: {}'.format(pp, cell_name) for cell_name in cell_names]

        elif dimensions == 'population':
            comp_names = ['C*_P{}'.format(pp)]
        else:
            raise ValueError("dimensions can only be 'cell' or 'population', but {} was given".format(dimensions))

        all_probes.append(disp_mat)
        compound_names.extend(comp_names)

    compound_names = np.asarray(compound_names)

    if dimensions == 'cell':
        # concatenates across first dimention i.e. cell/channel
        pop_pval = np.concatenate(all_probes, axis=0)
    elif dimensions == 'population':
        # stacs across new first dimention i.e. stimuli
        pop_pval = np.stack(all_probes, axis=0)
    else:
        raise ValueError("dimensions can only be 'cell' or 'population', but {} was given".format(dimensions))

    output = coll.namedtuple('population_pvals', 'matrix cell_names')

    return output(pop_pval, compound_names)

### complex plotting functions


def pseudopop_significance(signal, channels, probes=(1, 2, 3, 4), sign_fs=None, window=1, rolling=True, type='Kruskal',
                           consecutives=1, hist=False, bins=60, recache=False, signal_name=None):
    '''
    makes a summary plot of the significance(black dots) over time (x axis) for each combination of cell and
    *[contexts,...]-Probe (y axis).
    :param signal: a signal object with cpp epochs
    :param channels: channel index, cell name (or a list of the two previous). or kwd 'all' to define which channels to consider
    :param probes: list of ints, eache one corresponds to the identity of a vocalization used as probe.
    :sign_fs: sampling frequency at which perform the analysis, None uses de native sign_fs of the signal.
    :param sort: boolean. If True sort by last siginificant time bin.
    :param window: time window size, in time bins, over which calculate significant difference metrics
    :param rolling: boolean, If True, uses rolling window of stride 1. If False uses non overlaping yuxtaposed windows
    :param type: keyword defining what metric to use. 'Kruskal' for Kurscal Wallis,
    'Pearsons' for mean of pairwise correlation coefficient, 'MSD' for mean standard difference
    :consecutives: int number of consecutive significant time bins to consider "real" significance
    :param hist: Boolean, If True, draws a histogram of significance over time (cololapsing by cell-probe identity)
    :param bins: number of bins of the histogram
    :return: figure, axis
    '''
    # todo clean this function
    all_probes = list()
    compound_names = list()

    # handlese sign_fs
    sign_fs = hand._fs_handler(signal, sign_fs)

    # handles channel names
    channels = hand._channel_handler(signal, channels)

    # calculates dipersion pval for each set of contexts probe.
    for pp in probes:
        this_probe = r'\AC\d_P{}'.format(pp)
        disp_mat = signal_single_context_sigdif(signal, epoch_names=this_probe, channels=channels, fs=sign_fs,
                                                     window=window, rolling=rolling, type=type, recache=recache,
                                                     signal_name=signal_name)

        chan_idx = hand._channel_handler(signal, channels)
        cell_names = [name for nn, name in enumerate(signal.chans) if nn in chan_idx]
        comp_names = ['C*_P{}: {}'.format(pp, cell_name) for cell_name in cell_names]

        all_probes.append(disp_mat)
        compound_names.extend(comp_names)

    compound_names = np.asarray(compound_names)

    # concatenates across first dimention i.e. cell/channel
    pop_pval = np.concatenate(all_probes, axis=0)

    # defines significance "integration window"
    if isinstance(consecutives, int):
        consecutives = [consecutives]
    elif isinstance(consecutives, list):
        pass
    else:
        raise ValueError("consecutives should be a positive int or a list of ppositive ints")

    all_figs = list()
    diff_matrices = list()
    cell_orders = list()

    for cons in consecutives:
        pop_sign = _significance_criterion(pop_pval, axis=1, window=cons, threshold=0.01, comp='<=')  # array with shape

        times = np.arange(0, pop_sign.shape[1]) / sign_fs

        if hist is False:
            # set size of scatter markers
            scat_kwargs = {'s': 10}

            # organizes by last significant time for clarity
            def sort_by_last_significant_bin(unsorted):
                last_True = list()
                for cell in range(unsorted.shape[0]):
                    # find last significant point
                    idxs = np.where(unsorted[cell, :] == True)[0]
                    if idxs.size == 0:
                        idxs = 0
                    else:
                        idxs = np.max(idxs)
                    last_True.append(idxs)
                sort_idx = np.argsort(np.asarray(last_True))

                # initializes empty sorted array
                sorted_sign = np.empty(shape=unsorted.shape)
                for ii, ss in enumerate(sort_idx):
                    sorted_sign[ii, :] = pop_sign[ss, :]

                return sorted_sign, sort_idx

            sorted_sign, sort_idx = sort_by_last_significant_bin(pop_sign)
            sorted_names = compound_names[sort_idx]

            # rasters the sorted significances
            fig, ax = cplt._raster(times, sorted_sign, scatter_kws=scat_kwargs)
            ax.set_yticks(np.arange(0, sorted_sign.shape[0], 1))
            ax.set_yticklabels(sorted_names)

            fig.suptitle('metric: {}, window: {}, fs: {}, consecutives: {}'.format(type, window, sign_fs, cons))

        elif hist is True:
            _, sign_times = np.where(pop_sign == True)

            fig, ax = plt.subplots()
            ax.hist(sign_times, bins=bins)

        all_figs.append(fig)
        diff_matrices.append(sorted_sign)
        cell_orders.append(sorted_names)

    output = coll.namedtuple('Population_significance', 'figures diff_matrices cell_orders')

    return output(all_figs, diff_matrices, cell_orders)


def plot_single_context(signal, channels, epochs, sign_fs=None, raster_fs=None, psth_fs=None, window=1, rolling=False,
                        type='Kruskal', consecutives=1):
    '''
    calculates significant difference over time for the specified cells/channels and *[contexts,...]-probe (epochs),
    overlays gray vertical bars to hybrid plot (raster + PSTH) on time bins with significante difference between
    contexts
    :param channels: channel index, cell name (or a list of the two previos). or kwd 'all' to define which channels to consider
    :param epochs:  epoch name (str), regexp, list of epoch names, 'single', 'pair'. keywords 'single' and 'pair'
    correspond to all single vocalization, and pair of context probe vocalizations.
    :param window: time window size, in time bins, over which calculate significant difference metrics
    :param rolling: boolean, If True, uses rolling window of stride 1. If False uses non overlaping yuxtaposed windows
    :param type: keyword defining what metric to use. 'Kruskal' for Kurscal Wallis,
    :return: figure, axes
    '''
    # todo clean this function

    # calculates significant difference between different context across time
    disp_pval = signal_single_context_sigdif(signal, epoch_names=epochs, channels=channels, fs=sign_fs,
                                                  window=window,
                                                  rolling=rolling, type=type)

    all_figs = list()

    # defines significance "integration window"
    if isinstance(consecutives, int):
        consecutives = [consecutives]
    elif isinstance(consecutives, list):
        pass
    else:
        raise ValueError("consecutives should be a positive int or a list of ppositive ints")

    for con in consecutives:
        # defines significance, uses window size equal to time bin size
        significance = _significance_criterion(disp_pval, axis=1, window=con, threshold=0.01,
                                               comp='<=')  # array with shape

        # overlays significatn times on the raster and PSTH for the specified cells and context probe pairs
        scat_key = {'s': 5, 'alpha': 0.5}
        fig, axes = cplt.hybrid(signal, epoch_names=epochs, channels=channels, start=3, end=6, scatter_kws=scat_key,
                                significance=significance, raster_fs=raster_fs, psth_fs=psth_fs, sign_fs=sign_fs)

        fig.suptitle('metric: {}, window: {}, fs: {}, consecutives: {}'.format(type, window, sign_fs, con))

        all_figs.append(fig)

    return fig, axes


### test unit

def test_object():
    # single context params
    rep_num = 10
    cell_num = 2
    bin_num = 20

    cell_std = (0.25, 2)
    cont_means = (2, 5, 8)

    matrices = dict()

    for context, mean in enumerate(cont_means):
        matix = np.empty(
            [rep_num, cell_num, bin_num])  # shape: Repetitions x Channels x TimeBins
        for cell, std in enumerate(cell_std):
            for rep in range(rep_num):
                # generates a silence and signal strech of time, concatenates
                silence = np.zeros(int(bin_num / 2)) + np.random.normal(0, std, int(bin_num / 2))
                sound = np.empty(int(bin_num / 2));
                sound[:] = mean
                sound = sound + np.random.normal(mean, std, int(bin_num / 2))
                trial = np.concatenate([silence, sound], axis=0)

                matix[rep, cell, :] = trial

        matrices['C{}'.format(context)] = matix

    return matrices

# out = _single_cell_difsig(test_object(),window=5, rolling=True, type='Pearsons')
