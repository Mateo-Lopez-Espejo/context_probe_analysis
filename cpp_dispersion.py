import itertools as itt
import math

import numpy as np
import scipy.stats as sst
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances

import cpp_parameter_handlers as hand
### helper funtions
import cpp_plots as cplt


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


### helper dispersion functions

def _window_kruskal(working_slice):
    '''
    calculates kruskal wallis between contexts , binning by time,
    for an array of shape Repetition x Context x Time
    :param working_slice: 3d ndarray with dims Repetition x Context x Time
    :return: float pvalue
    '''
    # flattens the window time bin dimension into the Repetition dimension
    working_slice = np.swapaxes(working_slice, 0, 1)
    work_shape = working_slice.shape
    working_slice = working_slice.reshape(
        work_shape[0],
        work_shape[1] * work_shape[2], order='C')  # dimensions = Context x (Repetitions * window Time bin)

    # kruskal Wallis pvalue calculates simultaneously for all contexts
    try:
        kruscal = sst.kruskal(*working_slice)
        pval = kruscal.pvalue
    except:
        pval = np.nan

    return pval


def _window_pearsons(working_slice):
    '''
    calculates mean of the pairwise pearsons correlation of the PSTHs (i.e. collapsed repetitions) between contexts
    for an array of shape Repetition x Context x Time
    :param working_slice: 3d ndarray with dims Repetition x Context x Time
    :return: float pvalue
    '''

    def _working_slice_rval(working_slice):
        # input array should have shape Repetitions x Context x Time
        # calculates PSTH i.e. mean across repetitions
        psth = working_slice.mean(axis=0)  # dimentions Context x WindowTime
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
    obs_rval = _working_slice_rval(working_slice)

    # 2. shuffles across repetitions a contexts
    # collapses repetition and context together
    collapsed = working_slice.swapaxes(0, 2)  # maktes time the first axis, the only relevant to hold
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
        reshaped = shuffled.T.reshape(t, c, r)
        reshaped = reshaped.swapaxes(0, 1)
        # calculates pairwise r_value
        rval_floor[rep] = _working_slice_rval(reshaped)

    pvalue = (rval_floor > obs_rval).sum() / shuffle_n

    return pvalue


### base dispersion fucntions

def _single_cell_sigdif(matrices, channels='all', window=1, rolling=False, type='Kruskal'):
    '''
    given a dictionary of matrices (from signal.extract_epochs), calculates pvalue for a difference metric
    for the response to the different contexts of a stimuli (different epochs, each of the keywords in the dictionary).
    these calculations are done over time i.e. for each time bin

    :param matrices: a dictionary of matrices of dimensions Repetitions x Cells x Time. Each keywords corresponds to a different stimulus
    :param channels: the channels/cells to consider (second dimension of input matrices)
    :param window: window size in time bins over which to perform the calculations.
    :param rolling: boolena, wheather to use rolling windows or non overlapping yuxtaposed windows
    :param type: keyword defining what metric to use. 'Kruskal' for Kurscal Wallis,
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
            working_slice = windowed[:, cell, wind, :, :]  # array has dimensions Repetition x Context x window Time bin

            # selects betwee different metrics
            if type == 'Kruskal':
                pvalue = _window_kruskal(working_slice)

            if type == 'Pearsons':
                pvalue = _window_pearsons(working_slice)

            metric_over_time[cc, wind] = pvalue

    return metric_over_time


def _significance_criterion(pvalues, window=1, threshold=0.01, comp='<='):
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
    windowed = _into_windows(pvalues, window=window, axis=1, rolling=True, padding=np.nan)

    # which individual time bins are significant
    if comp == '<=':
        sign_bin = np.where(windowed <= threshold, True, False)
    elif comp == '>=':
        sign_bin = np.where(windowed >= threshold, True, False)
    else:
        raise ValueError(" only '<=' and '>=' defined")

    sign_window = np.all(sign_bin, axis=2)  # which windows contain only significant bins

    return sign_window


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


### signal wrapers

def signal_single_cell_sigdif(signal, epoch_names='single', channels='all', window=1, rolling=False,
                              type='Kruskal'):
    # handles epoch_names as standard
    epoch_names = hand._epoch_name_handler(signal, epoch_names)

    # handles channels/cells
    channels = hand._channel_handler(signal, channels)

    matrixes = signal.rasterize().extract_epochs(epoch_names)

    disp = _single_cell_sigdif(matrixes, channels=channels, window=window, rolling=rolling, type=type)

    return disp


### complex plotting functions

def population_significance(signal, channels, probes=(1, 2, 3, 4), sort=True, window=1, rolling=True, type='Kruskal',
                            hist=False, bins=60):
    '''
    makes a summary plot of the significance(black dots) over time (x axis) for each combination of cell and
    *[contexts,...]-Probe (y axis).
    :param signal: a signal object with cpp epochs
    :param channels: channel index, cell name (or a list of the two previos). or kwd 'all' to define which channels to consider
    :param probes: list of ints, eache one corresponds to the identity of a vocalization used as probe.
    :param sort: boolean. If True sort by last siginificant time bin.
    :param window: time window size, in time bins, over which calculate significant difference metrics
    :param rolling: boolean, If True, uses rolling window of stride 1. If False uses non overlaping yuxtaposed windows
    :param type: keyword defining what metric to use. 'Kruskal' for Kurscal Wallis,
    'Pearsons' for mean of pairwise correlation coefficient.
    :param hist: Boolean, If True, draws a histogram of significance over time (cololapsing by cell-probe identity)
    :param bins: number of bins of the histogram
    :return: figure, axis
    '''
    # todo clean this function
    all_probes = list()
    compound_names = list()

    # calculates dipersion pval for eache set of contexts probe.
    for pp in probes:
        this_probe = r'\AC\d_P{}'.format(pp)
        disp_mat = signal_single_cell_sigdif(signal, epoch_names=this_probe, channels=channels,
                                             window=window, rolling=rolling, type=type)

        chan_idx = hand._channel_handler(signal, channels)
        cell_names = [name for nn, name in enumerate(signal.chans) if nn in chan_idx]
        comp_names = ['C*_P{}: {}'.format(pp, cell_name) for cell_name in cell_names]

        all_probes.append(disp_mat)
        compound_names.extend(comp_names)

    compound_names = np.asarray(compound_names)

    # concatenates across first dimention i.e. cell/channel
    pop_pval = np.concatenate(all_probes, axis=0)

    # defines significance, uses window size equal to time bin size
    pop_sign = _significance_criterion(pop_pval, window=1, threshold=0.01, comp='<=')  # array with shape

    times = np.arange(0, pop_sign.shape[1]) / signal.fs

    if hist is False:
        # set size of scatter markers
        scat_kwargs = {'s': 10}
        # raster significance, unsorted
        if sort is False:
            fig, ax = cplt._raster(times, pop_sign, scatter_kws=scat_kwargs)
            ax.set_yticks(np.arange(0, pop_sign.shape[0], 1))
            ax.set_yticklabels(compound_names)

        elif sort is True:
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

    elif hist is True:
        _, sign_times = np.where(pop_sign == True)

        fig, ax = plt.subplots()
        ax.hist(sign_times, bins=bins)

    return fig, ax


def plot_single(signal, channels, epochs, window=1, rolling=False, type='Kruskal'):
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
    disp_pval = signal_single_cell_sigdif(signal, epoch_names=epochs, channels=channels, window=window,
                                          rolling=rolling, type=type)

    # defines significance, uses window size equal to time bin size
    significance = _significance_criterion(disp_pval, window=1, threshold=0.01, comp='<=')  # array with shape

    # overlays significatn times on the raster and PSTH for the specified cells and context probe pairs
    scat_key = {'s': 5, 'alpha': 0.5}
    fig, axes = cplt.hybrid(signal, epoch_names=epochs, channels=channels, start=3, end=6, scatter_kws=scat_key,
                            significance=significance)

    return fig, axes


### test unit

def test_object():
    # single context params
    rep_num = 10
    cell_num = 2
    bin_num = 20

    cell_std = (0.25, 2)
    cont_means = (2 ,5, 8)

    matrices = dict()

    for context, mean in enumerate(cont_means):
        matix = np.empty(
            [rep_num, cell_num, bin_num])  # shape: Repetitions x Channels x TimeBins
        for cell, std in enumerate(cell_std):
            for rep in range(rep_num):
                # generates a silence and signal strech of time, concatenates
                silence = np.zeros(int(bin_num/2)) + np.random.normal(0, std, int(bin_num/2))
                sound = np.empty(int(bin_num/2)); sound[:] = mean
                sound = sound + np.random.normal(mean, std, int(bin_num/2))
                trial = np.concatenate([silence, sound], axis=0)

                matix[rep, cell, :] = trial

        matrices['C{}'.format(context)] = matix

    return matrices