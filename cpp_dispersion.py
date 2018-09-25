import numpy as np
import scipy.stats as sst
from sklearn.metrics.pairwise import pairwise_distances

import cpp_parameter_handlers as hand

### helper funtions

def _into_windows(array, window, axis = -1, rolling=True, padding=np.nan,):
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
        newshape.append(window) # just add a new dimention of window length and the end
        windowed = np.empty(newshape)

        for ii in range(old_shape[-1]):
            windowed[..., ii, :] = padded[..., ii:ii + window]

    elif rolling == False:
        old_ax_len = int(pad_shape[-1]/window) # windowed dimention divided by window len
        newshape = pad_shape
        newshape[-1] = old_ax_len # modifies old dimention
        newshape.append(window) # add new dimention i.e. elements along window.
        windowed = padded.reshape(newshape, order='C') # C order, changes las index the fastest.


    # returs to the "original" dimention order, where the windowed dimention becomes the window dimention i.e. window number
    # and the last dimension becomes the values corresponding to the selected dimnesion, along each window.
    windowed = windowed.swapaxes(axis, -2) # swaps the original last dimension(-2) back in place

    return windowed


def _into_rolling_windows(array, window, axis, padding=np.nan): # todo deprecate

    '''
    makes a 2d array into a 3d array containing a colection of rolled windows in time
    :param array: 2D array, with dimentions Channels x Time bins
    :param window: window size in time bins
    :param padding: value to fill the end of the rolling windows, nan is default
    :return: 3d array with dimetions Channels x Windows x Time bins in window
    '''

    padded = np.empty([array.shape[0], array.shape[1] + window - 1])
    padded[:] = padding
    padded[:array.shape[0], :array.shape[1]] = array

    windowed = np.empty([array.shape[0], array.shape[1], window])

    for ii in range(array.shape[1]):
        windowed[:, ii, :] = padded[:, ii:ii + window]

    return windowed


def _into_static_windows(array, window, padding=np.nan): #todo deprecate
    '''
    makes a 2d array into a 3d array containing a collection of succesive, non overlaping windows in time
    :param array: 2d array, with dimentions Channels x Time bins
    :param window: window size in time bins
    :param padding: value to fill the last windows, nan is default
    :return: 3d array with dimentions Channels x Windows x Time bins in window
    '''

    extra_len = int((window * np.ceil(array.shape[1]/window)) - array.shape[1])

    padded = np.empty([array.shape[0], array.shape[1] + extra_len])
    padded[:] = padding
    padded[:array.shape[0], :array.shape[1]] = array

    windowed = padded.reshape(array.shape[0], int(padded.shape[1]/window), window)

    return windowed


### base dispersion fucntions

def _single_cell_dispersion(matrixes, channels='all', window=1, rolling=False):
    '''
    given a dictionary of matrices (from signal.extract_epochs), calculates pvalue for a Kruskal Wallis considering
    the response to the different stimuli (different epochs, each of the keywords in the dictionary)
    . these calculations are done over time i.e. for each time bin
    :param matrix: a dictionary of matrices of dimensions Repetitions x Cells x Time.
                   Each keywords corresponds to a different stimulus
    :channels: the channels/cells to consider (second dimension of input matrices)
    :return: an array of shape Cell x Time, of pvalues for each cell across time.
    '''
    # stacks all matrixes (different vocalizations) across new axis, then selects the desired cell
    full_mat = np.stack(matrixes.values(), axis=3) # shape: Repetitions x Channels x TimeBins x ContextStimuli

    # handles channel keywords
    channels = hand._channel_handler(full_mat[..., 0], channels)

    # generates the windowed array

    windowed = _into_windows(full_mat, window=window, axis=2, rolling=rolling)


    # initializes result matrix with shape C x T where C is cell and T is time or TimeWindow
    shape = windowed.shape
    kruscal_over_time = np.zeros([len(channels), shape[2]])  # empty array of dimentions Cells x Time

    # iterates over cell
    for cc, cell in enumerate(channels):
        # iterates over time window
        for time in range(shape[2]):
            working_slice = windowed[:, cell, time, :,:]  # array has dimentions Repetition x Context x winndow Time bin
            # flattens the window time bin dimension into the Repetition dimension
            working_slice = np.swapaxes(working_slice, 0, 1)
            work_shape = working_slice.shape
            working_slice = working_slice.reshape(work_shape[0], work_shape[1] * work_shape[2])

            try:
                kruscal = sst.kruskal(*working_slice)
                pval = kruscal.pvalue
            except:
                pval = np.nan
            kruscal_over_time[cc, time] = pval

    return kruscal_over_time


def _significance_criterion(pvalues, window=1, alpha=0.01):
    '''
    acording to Asari and sador, to determine significance of a contextual effect, and to avoid false possitive
    due to multiple comparisons, significant differences are only acepted if there are streches of consecutive time bins
    all with significance < 0.01. takes an array of pvalues and returns a boolean vector of significance
    acording to an alpha threshold an a window size
    :param pvalues: 2d array of dimentions C x T where C are cells/channels and T are time bins
    :param window: rolling window sizes in number of time bins, default 1 i.e time window = bin size
    :param alpha: certainty threshold, by default 0.01
    :return: boolean array of the same dimentions of pvalues array
    '''

    # windowed = _into_rolling_windows(pvalues, window, padding=np.nan)
    windowed = _into_windows(pvalues, window=window, axis=1, rolling=True, padding=np.nan)

    sign_bin = np.where(windowed <= alpha, True, False)  # which individual time bins are significant

    sign_window = np.all(sign_bin, axis=2)  # which windows contain only significant bins

    return sign_window


def _sig_bin_to_time(sign_window, window, fs):
    # takes a boolean matrix of significance, the size of the window and the sampling frequency an transforms into a
    # array of times describing the start and end of streches of significance

    start_times = list()
    end_times = list()

    for cc in range(sign_window.shape[0]):  # iterates over the channels/cells
        bin_ind = np.where(sign_window[cc, :] == True)[0]
        start = bin_ind / fs  # thise indexing takes out the array from the tupple
        end = start + (window / fs)
        start_times.append(start)
        end_times.append(end)

    return start_times, end_times


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

def signal_single_cell_dispersion(signal, epoch_names='single', channels='all', window_kws=None):

    # window keywords
    window_kws = {} if window_kws is None else window_kws


    # handles epoch_names as standard
    epoch_names = hand._epoch_name_handler(signal, epoch_names)

    # handles channels/cells
    channels = hand._channel_handler(signal, channels)

    matrixes = signal.rasterize().extract_epochs(epoch_names)

    disp = _single_cell_dispersion(matrixes, channels=channels, **window_kws)

    return disp


### pipeline wrappers

