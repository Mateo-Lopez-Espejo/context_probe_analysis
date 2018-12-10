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

### defines all named tuples for pickling
dispersion_over_time = coll.namedtuple('dispersion_over_time', 'metric pvalue')
population_dispersion = coll.namedtuple('population_dispersion', 'matrix good_unit_names')


### helper funtions

def _into_windows(array, window, axis=-1, rolling=True, padding=np.nan, ):
    '''
    I am so proud of this function. Takes an nd array, of N dimensions and generates an array of N+1 of windows across
    the selected dimension. The selected dimension becomes window number, and the new last dimension becomes the original
    dimension units across the window length.
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
        metric, pvalue = sst.kruskal(*working_window)

    except:
        metric = pvalue = np.nan

    return metric, pvalue


def _window_STD(working_window):
    '''
    calculates standard deviation of the PSTH between contexts , binning by time with the mean,
    for an array of shape Repetition x Context x Time
    :param working_window: 3d ndarray with dims Repetition x Context x Time
    :return: float pvalue
    '''

    # calculates the PSTH
    psth = np.mean(working_window, axis=0)  # PSTH with shape Context x Time

    # calculates the standard deviation across contexts
    ctx_std = np.std(psth, axis=0)  # std of PSTHs with shape Time

    # gets the mean acroos time
    metric = np.mean(ctx_std)
    pvalue = np.nan

    return metric, pvalue


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

    metric = obs_rval

    return metric, pvalue


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

    metric = obs_msd

    return metric, pvalue


def _window_euclidean(working_window):
    '''
        calculates mean of the pairwise euclidean distance of the PSTHs (i.e. collapsed repetitions) between contexts
        for an array of shape Repetition x Context x Time
        :param working_window: 3d ndarray with dims Repetition x Context x Time
        :return: float pvalue
        '''

    def _working_window_euclidean(working_window):
        # input array should have shape Repetitions x Context x Time
        # calculates PSTH i.e. mean across repetitions
        psth = working_window.mean(axis=0)  # dimentions Context x WindowTime
        # initializes array to hold the calculated metric for all context pairs combinations
        combs = int(math.factorial(psth.shape[0]) / (2 * math.factorial(psth.shape[0] - 2)))
        euc_values = np.empty(combs)
        # iterates over every pair of contexts
        for ctx, (ctx1, ctx2) in enumerate(itt.combinations(range(psth.shape[0]), 2)):
            # calculates an holds the euclidean distance
            euc = (np.sqrt((psth[ctx1, :] - psth[ctx2, :]) ** 2))

            # takes the mean across time
            t_mean = np.nanmean(euc)

            euc_values[ctx] = t_mean

        # calculates the mean of the pairwise distances.
        pair_euc_mean = euc_values.mean()
        return pair_euc_mean

    # 1. calculates the mean of the pairwise correlation coefficient between all differente contexts
    obs_msd = _working_window_euclidean(working_window)

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
        msd_floor[rep] = _working_window_euclidean(reshaped)

    # pvalue = (msd_floor > obs_msd).sum() / shuffle_n

    if obs_msd > msd_floor.mean():
        pvalue = (msd_floor > obs_msd).sum() / shuffle_n
    elif obs_msd < msd_floor.mean():
        pvalue = (msd_floor < obs_msd).sum() / shuffle_n
    else:
        pvalue = 1

    metric = obs_msd

    return metric, pvalue


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

    metric = obs_msd

    return metric, pvalue


def _window_ndim_STD(working_window):
    '''
    todo this function is shit. I cannot figure out a proper manner of considering standard deviation across the dimentions of context, and cells
    calculates standard deviation of the PSTH between contexts , binning by time with the mean,
    for an array of shape Repetition x Context x Time
    :param working_window: 3d ndarray with dims Repetition x Context x Time
    :return: float pvalue
    '''

    # input array should have shape Repetition x Unit x Context x WindowTimeBin
    # calculates PSTH
    psth = np.mean(working_window, axis=0)  # PSTH with shape Unit x Context x Time

    # calculates the cell-specific context-driven dispersion
    cell_std = np.std(psth, axis=1)  # cell STD with shape Unit x Time

    # mean of the std across cells, and bins by meaning in time
    metric = np.mean(cell_std)

    pvalue = np.nan

    return metric, pvalue


def _window_ndim_euclidean(working_window):
    '''
    calculates mean of the pairwise Mean Standard Difference of the PSTHs (i.e. collapsed repetitions) between contexts
    for an array of shape Repetition x Unit x Context x WindowTimeBin
    :param working_window: 4D array has dimensions Repetition x Unit x Context x WindowTimeBin
    :return: float pvalue
    '''

    def _working_window_ndim_euclidean(working_window):
        # input array should have shape Repetition x Unit x Context x WindowTimeBin
        # calculates PSTH i.e. mean across repetitions
        psth = working_window.mean(axis=0)  # dimentions Unit x Context x WindowTime
        # initializes array to hold the calculated metric for all context pairs combinations
        combs = int(math.factorial(psth.shape[1]) / (2 * math.factorial(psth.shape[1] - 2)))
        msd_values = np.empty(combs)
        # iterates over every pair of contexts
        for ctx, (ctx1, ctx2) in enumerate(itt.combinations(range(psth.shape[1]), 2)):
            # compares the psth difference for each Unit/state-dimension
            unit_sqr_dif = np.sqrt(np.sum((psth[:, ctx1, :] - psth[:, ctx2, :]) ** 2, axis=0))
            # means across time
            msd_values[ctx] = np.nanmean(unit_sqr_dif, axis=0)

        mean_r = msd_values.mean()
        return mean_r

    # 1. calculates the mean of the pairwise correlation coefficient between all differente contexts
    obs_euc = _working_window_ndim_euclidean(working_window)

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
        msd_floor[rep] = _working_window_ndim_euclidean(reshaped)

    # pvalue = (msd_floor > obs_euc).sum() / shuffle_n

    if obs_euc > msd_floor.mean():
        pvalue = (msd_floor > obs_euc).sum() / shuffle_n
    elif obs_euc < msd_floor.mean():
        pvalue = (msd_floor < obs_euc).sum() / shuffle_n
    else:
        pvalue = 1

    metric = obs_euc

    return metric, pvalue


### population single trial dispersion calculations

def _pairwise_single_trial_ndim_euclidean(matrix0, matrix1, zero_handling=None):
    '''
    base single trial distance calculation function. takes two matrixes with the shape
    and calculates the n-dimentional euclidean distance betwee each repetition pair (a,b) where a comes from matrix0
    and b from matrix1. returns an array of these distances over time

    :param matrix0: 3D matrix withe shape Repetitions x Channels x TimeBins
    :param matrix1: 3D matrix withe shape Repetitions x Channels x TimeBins
    :zero_handling: how to treat the distance between 0 and 0
    :return: 2D matrix with shape PairDistnace x TimeBin
    '''
    # checks for consisten shape between the Channels and Timebins of the matrixes
    if matrix0.shape[1:] != matrix1.shape[1:]:
        raise ValueError('matrixes with unconsisten shapes: {} and {}'.format(matrix0.shape, matrix1.shape))

    # initializes distance array of shape Combinations x TimeBins
    distance_array = np.empty([matrix0.shape[0] * matrix1.shape[0], matrix0.shape[2]])
    distance_array1 = np.empty([matrix0.shape[0] * matrix1.shape[0], matrix0.shape[2]])

    # iterates over eache pair of single trial responses
    for ee, (ii, jj) in enumerate(itt.product(range(matrix0.shape[0]), range(matrix1.shape[0]))):

        x = matrix0[ii, :, :].T
        y = matrix1[jj, :, :].T
        x_min_y = x - y
        distance_array[ee, :] = np.sqrt(np.einsum('ij,ij->i', x_min_y, x_min_y))

    return distance_array



### dispersion summary metrics

def disp_exp_decay(matrix, start=None, prior=None, C=None, axis=None):
    # todo define axis parameter effect
    if start is None:
        start = 0
    # todo define what to do with prior == None

    # calculates the y intercept as the mean dispersione between Prior and start
    if C is None:
        if prior is None:
            C = 0
        else:
            C = np.mean(matrix[start-prior:start])
    elif isinstance(C, (float, int)):
        pass

    t = np.arange(0, len(matrix[start:]))
    y = matrix[start:]

    def model_func(t, A, K, C):
        return A * np.exp(K * t) + C

    y1 = y - C
    y1 = np.log(y1)
    K, A_log = np.polyfit(t, y1, 1)
    A = np.exp(A_log)

    fit_y = model_func(t, A, K, C)

    # defines first half as nan, and prepend to fit_y
    y_context = np.empty(len(matrix[:start]))
    y_context[:] = np.nan

    full_fit = np.concatenate([y_context, fit_y])

    return full_fit, A, K, C


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
    pvalue_over_time = np.zeros([len(channels), shape[2]])  # empty array of dimentions Cells x Time

    # iterates over cell
    for cc, cell in enumerate(channels):
        # iterates over time window
        for wind in range(shape[2]):
            working_window = windowed[:, cell, wind, :,
                             :]  # array has dimensions Repetition x Context x window Time bin

            # selects betwee different metrics
            if type == 'Kruskal':
                metric, pvalue = _window_kruskal(working_window)

            elif type == 'Pearsons':
                if window == 1: raise ValueError('Pearsons correlation requieres window of size > 1')
                metric, pvalue = _window_pearsons(working_window)

            elif type == 'MSD':
                metric, pvalue = _window_MSD(working_window)

            elif type == 'STD':
                metric, pvalue = _window_MSD(working_window)

            elif type == 'Euclidean':
                metric, pvalue = _window_euclidean(working_window)

            else:
                raise ValueError('keyword {} not suported'.format(type))

            metric_over_time[cc, wind] = metric
            pvalue_over_time[cc, wind] = pvalue

    return dispersion_over_time(metric_over_time, pvalue_over_time)


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
    channels = hand._channel_handler(full_mat[..., 0], channels)  # uses a single cotext matrix to check channel format.

    # takes only the channels to be used
    chan_mat = np.take(full_mat, channels, axis=1)

    # generates the windowed array
    # shape REpetition x Channels x Window x ContextStimuli x WindowTimeBin.
    windowed = _into_windows(chan_mat, window=window, axis=2, rolling=rolling)

    # initializes result matrix with shape T where T is time or TimeWindow
    shape = windowed.shape
    metric_over_time = np.zeros(shape[2])
    pvalue_over_time = np.zeros(shape[2])

    # iterates over time window
    for wind in range(shape[2]):
        working_window = windowed[:, :, wind, :, :]  # array has dimensions Repetition x Cell x Context x WindowTimeBin

        if type == 'MSD':
            metric, pvalue = _window_ndim_MSD(working_window)

        elif type == 'Euclidean':
            metric, pvalue = _window_ndim_euclidean(working_window)

        else:
            raise ValueError('keyword {} not suported'.format(type))

        metric_over_time[wind] = metric
        pvalue_over_time[wind] = pvalue

    return dispersion_over_time(metric_over_time, pvalue_over_time)


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

def signal_single_context_sigdif(signal, epoch_names='single', channels='all', dimensions='cell', fs=None, window=1,
                                 rolling=False, type='Kruskal'):

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
        dispersion_over_time = _population_difsig(matrixes, channels=channels, window=window, rolling=rolling,
                                                  type=type)

    elif dimensions == 'cell':
        dispersion_over_time = _single_cell_difsig(matrixes, channels=channels, window=window, rolling=rolling,
                                                   type=type)

    else:
        raise ValueError("dimensions can only be 'cell' or 'population', but '{}' was given".format(dimensions))

    return dispersion_over_time


def signal_all_context_sigdif(signal, channels, signal_name, probes=(1, 2, 3, 4), dimensions='cell', sign_fs=None, window=1,
                              rolling=True, type='Kruskal', recache=False, value='pvalue'):
    '''
    todo finish documentation
    calculates the difference pvalue across all cells in all probes
    :param signal: cpp Singal object
    :param channels: channel index, cell name (or a list of the two previous). or kwd 'all' to define which channels to consider
    :signal_name: unique signal name for proper caching.
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

        disp_func_args = {'signal': signal, 'epoch_names': this_probe, 'channels': channels,
                          'dimensions': dimensions,
                          'fs': sign_fs,
                          'window': window,
                          'rolling': rolling, 'type': type}

        dispersion_over_time_path = cch.make_cache(function=signal_single_context_sigdif, func_args=disp_func_args,
                                                   classobj_name=signal_name, recache=recache,
                                                   cache_folder='/home/mateo/mycache/single_probe_disp')

        dispersion_over_time = cch.get_cache(dispersion_over_time_path)



        if dimensions == 'cell':
            chan_idx = hand._channel_handler(signal, channels)  # heterogeneous "channels" value to indexes
            cell_names = [name for nn, name in enumerate(signal.chans) if nn in chan_idx]
            comp_names = ['C*_P{}: {}'.format(pp, cell_name) for cell_name in cell_names]

        elif dimensions == 'population':
            comp_names = ['C*_P{}'.format(pp)]
        else:
            raise ValueError("dimensions can only be 'cell' or 'population', but {} was given".format(dimensions))

        if value == 'metric':
            all_probes.append(dispersion_over_time.metric)
        elif value == 'pvalue':
            all_probes.append(dispersion_over_time.pvalue)
        else:
            raise ValueError("parameter 'value' can only be 'metric' or 'pvalue'. {} was given".format(value))

        compound_names.extend(comp_names)

    compound_names = np.asarray(compound_names)

    if dimensions == 'cell':
        # concatenates across first dimention i.e. cell/channel
        disp_val = np.concatenate(all_probes, axis=0)
    elif dimensions == 'population':
        # stacs across new first dimention i.e. stimuli
        disp_val = np.stack(all_probes, axis=0)
    else:
        raise ValueError("dimensions can only be 'cell' or 'population', but {} was given".format(dimensions))

    return population_dispersion(disp_val, compound_names)


### complex plotting functions


def pseudopop_significance(signal, channels, signal_name, probes=(1, 2, 3, 4), sign_fs=None, window=1, rolling=True, type='Kruskal',
                           consecutives=1, hist=False, bins=60, recache=False, value='pvalue'):
    '''
    makes a summary plot of the significance(black dots) over time (x axis) for each combination of cell and
    *[contexts,...]-Probe (y axis).
    :param signal: a signal object with cpp epochs
    :param channels: channel index, cell name (or a list of the two previous). or kwd 'all' to define which channels to consider
    :signal_name: unique signal name, for proper caching purpose
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

    # calculates dipersion pval for each set of contexts probe.
    for pp in probes:
        this_probe = r'\AC\d_P{}'.format(pp)

        disp_func_args = {'signal': signal, 'epoch_names': this_probe, 'channels': channels,
                          'dimensions': 'cell',
                          'fs': sign_fs,
                          'window': window,
                          'rolling': rolling, 'type': type}

        dispersion_over_time_path = cch.make_cache(function=signal_single_context_sigdif, func_args=disp_func_args,
                                                   classobj_name=signal_name, recache=recache,
                                                   cache_folder='/home/mateo/mycache/single_probe_disp')
        dispersion_over_time = cch.get_cache(dispersion_over_time_path)

        chan_idx = hand._channel_handler(signal, channels)
        cell_names = [name for nn, name in enumerate(signal.chans) if nn in chan_idx]
        comp_names = ['C*_P{}: {}'.format(pp, cell_name) for cell_name in cell_names]

        if value == 'metric':
            all_probes.append(dispersion_over_time.metric)
        elif value == 'pvalue':
            all_probes.append(dispersion_over_time.pvalue)
        else:
            raise ValueError("parameter 'value' can only be 'metric' or 'pvalue'. {} was given".format(value))

        compound_names.extend(comp_names)

    compound_names = np.asarray(compound_names)

    # concatenates across first dimention i.e. cell/channel, final shape is (Context x Cell) x Time
    population_dispersion = np.concatenate(all_probes, axis=0)

    # defines sorting function
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

    if value == 'pvalue':
        # defines significance "integration window"
        if isinstance(consecutives, int):
            consecutives = [consecutives]
        elif isinstance(consecutives, list):
            pass
        else:
            raise ValueError("consecutives should be a positive int or a list of positive ints")

        all_figs = list()
        diff_matrices = list()
        cell_orders = list()

        for cons in consecutives:
            pop_sign = _significance_criterion(population_dispersion, axis=1, window=cons, threshold=0.01,
                                               comp='<=')  # array with shape

            times = np.arange(0, pop_sign.shape[1]) / sign_fs

            if hist is False:
                # set size of scatter markers
                scat_kwargs = {'s': 10}

                # organizes by last significant time for clarity
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

    elif value == 'metric':

        diff_matrices = [population_dispersion]
        cell_orders = [compound_names]
        times = np.arange(0, population_dispersion.shape[1]) / sign_fs
        fig, ax = cplt._heatmap(times, population_dispersion)
        ll_figs = [fig]

    output = coll.namedtuple('Population_significance', 'figures diff_matrices cell_orders')

    return output(all_figs, diff_matrices, cell_orders)


def plot_single_probe(signal, signal_name, channels, epochs, sign_fs=None, raster_fs=None, psth_fs=None, window=1, rolling=False,
                      type='Kruskal', consecutives=1, value='pvalue'):
    '''
    calculates significant difference over time for the specified cells/channels and *[contexts,...]-probe (epochs),
    overlays gray vertical bars to hybrid plot (raster + PSTH) on time bins with significante difference between
    contexts
    :param channels: channel index, cell name (or a list of the two previos). or kwd 'all' to define which channels to consider
    :param epochs:  epoch name (str), regexp, list of epoch names, 'single', 'pair'. keywords 'single' and 'pair'
    correspond to all single vocalization, and pair of context probe vocalizations.
    :param window: time window size, in time bins, over which calculate significant difference metrics
    :param rolling: boolean, If True, uses rolling window of stride 1. If False uses non overlapping juxtaposed windows
    :param type: keyword defining what metric to use. 'Kruskal' for Kurskal Wallis,
    :consecutives: int, [int...], number of consecutive significant time windows to consider overall significance. works only when ploting pvalues
    :value: keyword 'metric', plots the dispersion metric; 'pvalue' plots the corresponding pvalue if possible
    :return: figure, axes
    '''
    # todo clean this function

    # calculates significant difference between different context across time
    disp_func_args = {'signal':signal, 'epoch_names':epochs, 'channels':channels,
                      'dimensions':'cell',
                      'fs':sign_fs,
                      'window':window,
                      'rolling':rolling, 'type':type}

    dispersion_over_time_path = cch.make_cache(function=signal_single_context_sigdif, func_args=disp_func_args,
                                               classobj_name=signal_name,
                                               cache_folder='/home/mateo/mycache/single_probe_disp')

    dispersion_over_time = cch.get_cache(dispersion_over_time_path)


    all_figs = list()
    if value == 'pvalue':
        # defines significance "integration window"
        if isinstance(consecutives, int):
            consecutives = [consecutives]
        elif isinstance(consecutives, list):
            pass
        else:
            raise ValueError("consecutives should be a positive int or a list of ppositive ints")

        for con in consecutives:
            # defines significance, uses window size equal to time bin size
            significance = _significance_criterion(dispersion_over_time.pvalue, axis=1, window=con, threshold=0.01,
                                                   comp='<=')  # array with shape Cell x Time

            # overlays significatn times on the raster and PSTH for the specified cells and context probe pairs
            scat_key = {'s': 5, 'alpha': 0.5}
            fig, axes = cplt.hybrid(signal, epoch_names=epochs, channels=channels, start=3, end=6, scatter_kws=scat_key,
                                    significance=significance, raster_fs=raster_fs, psth_fs=psth_fs, sign_fs=sign_fs)

            fig.suptitle('metric: {}, window: {}, fs: {}, consecutives: {}'.format(type, window, sign_fs, con))

            all_figs.append(fig)

    elif value == 'metric':
        raise NotImplementedError('ups')

    return fig, axes


