import math
import pathlib as pl
from configparser import ConfigParser
import warnings

import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import get_test_data
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.ndimage.filters as sf
import scipy.signal as ssig
import scipy.stats as sst
from scipy.stats import linregress

from cpp_parameter_handlers import _epoch_name_handler, _channel_handler, _fs_handler
import fits as fts
from nems.signal import PointProcess


config = ConfigParser()
if pl.Path('../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../context_probe_analysis/config/settings.ini'))
elif pl.Path('../../../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../../../context_probe_analysis/config/settings.ini'))
else:
    raise FileNotFoundError('config file coluld not be foud')



# todo redo the distribution of axes in figures for psth, there is not purpose in plotig multiple signals as different
# rows, rather the subplotse should correspond to individual cells, and elements within each axis should correspond to
# responses to different stimuli ...

### helper functions

def subplots_sqr(n_subplots, **sp_kwargs):
    '''
    makes an square array of subplots depending on the total number of subplots desired
    :param n_subplots:
    :param sp_kwargs:
    :return:
    '''

    rows = int(np.ceil(math.sqrt(n_subplots)))
    cols = int(np.floor(math.sqrt(n_subplots)))

    if rows*cols < n_subplots:
        rows = rows+1

    defaults = {'sharex':False, 'sharey':False, 'squeeze':False}
    for key, arg in defaults.items(): sp_kwargs.setdefault(key, arg)

    fig, axes = plt.subplots(rows, cols, **sp_kwargs)

    axes = np.ravel(axes)

    return fig, axes


def _sig_bin_to_time(sign_window, window, fs, unit_overlaping=True):
    '''
    takes a boolean array of significance with shape Cell/Channle x Timebins, and returns the time values of start and end of
    the significant bins. This are organized in two lists start_time and end_time, each of len equal to cell number
    containing 1d arrays with time in seconds.
    None, the output are lists of 1d arrays instead of 2d arrays because of the non uniform lenth of the 1d arrys
    i.e. the number of significant bins
    :param sign_window:
    :param window:
    :param fs:
    :param unit_overlaping:
    :return:
    '''
    # takes a boolean matrix of significance, the size of the window and the sampling frequency an transforms into two
    # arrays of times describing the start and end of streches of significance

    start_times = list()
    end_times = list()

    if sign_window.ndim == 1:
        sign_window = np.expand_dims(sign_window, axis=0)
    elif sign_window.ndim == 2:
        pass
    else:
        raise ValueError('sign_window has too many dimensions ')

    for cc in range(sign_window.shape[0]):  # iterates over the channels/cells
        bin_ind = np.where(sign_window[cc, :] == True)[0]  # thise indexing takes out the array from the tupple
        start = bin_ind / fs
        start.sort()
        end = start + (window / fs)

        if unit_overlaping is True:
            i = 0
            n = len(start)
            u_start = list()
            u_end = list()
            while i < n:
                s, e = start[i], end[i]
                i += 1
                while (i < n) and ((e > start[i]) or (math.isclose(e, start[i], abs_tol=1e-10))):
                    e = end[i]
                    i += 1
                u_start.append(s)
                u_end.append(e)

            start = np.asarray(u_start)
            end = np.asarray(u_end)

        start_times.append(start)
        end_times.append(end)

    return start_times, end_times


### base plotting functions


def _raster(times, values, y_offset=None, y_range=None, ax=None, scatter_kws=None):
    '''
    Plots a raster with one line for each pair of
    time and value vectors.
    Lines will be auto-colored according to matplotlib defaults.

    times : Array with shape T where T is time bins in seconds
    values : Array with shape R x T : where R is repetitions, and T is time. the dimension of T must agree with that of
             T in times.
    xlabel : str
    ylabel : str
    scatter_kws : pass-through arguments for plt.scatter
    '''

    if ax == None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x = values.copy()
    x = x[np.isfinite(x[:, 0]), :]  # discards channels with non numeric values

    i, j = np.where(x > 0)  # finds "spikes"

    if y_range != None:
        # sets the values of i in between the range defiend by y_range
        i = (i * (y_range[1] - y_range[0]) / np.max(i)) + y_range[0]

    if y_offset != None and y_range == None:
        i += y_offset

    if times is not None:
        t = times[j]
    else:
        t = j

    # updates kwargs
    scatter_kws = {} if scatter_kws is None else scatter_kws
    defaults = {'s': 1, 'color': 'black', 'marker': '.'}
    for key, arg in defaults.items(): scatter_kws.setdefault(key, arg)

    # plots
    ax.scatter(t, i, **scatter_kws)

    return fig, ax


def _heatmap(times, values, y_offset=None, ax=None, imshow_kws=None):
    '''
        Plots a heatmap with one line for each value vector

        times : Array with shape T where T is time bins in seconds
        values : Array with shape R x T : where R is repetitions, and T is time. the dimensio of T must agree with that of
                 T in times.
        imshow_kws : pass-through arguments for plt.imshow
        '''
    # ToDo check thise works as expected

    if ax == None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x = values.copy()
    x = x[np.isfinite(x[:, 0]), :]  # discards channels with non numeric values

    i, j = np.where(x > 0)  # finds "spikes"

    if y_offset != None:
        i += y_offset

    if times is not None:
        t = times[j]
    else:
        t = j

    # defines bounds of image for compatibility with other plot types

    left = 0
    right = times[-1]
    bottom = values.shape(0)
    top = 0

    # updates kwargs
    imshow_kws = {} if imshow_kws is None else imshow_kws
    defaults = {'origin': 'upper',
                'extent': (left, right, bottom, top)}
    for key, arg in defaults.items(): imshow_kws.setdefault(key, arg)

    # plots
    ax.imshow(values, **imshow_kws)

    return fig, ax


def _PSTH(times, values, start=None, end=None, ax=None, ci=False, y_offset=None, y_scaling=None, plot_kws=None):
    '''
        Base function for plotting a PSTH from a 2d matix with dimentions R x T where R is the repetitions
        and T is time. This matrix usually comes from "extract_epochs" afeter selecting a specific channel/cell
        :times: Array with shape T where T is time bins in seconds
        :values: Array with shape R x T : where R is repetitions, and T is time. the dimention of T must agree with that of T in times.
        :param start: the time of the sound onset in seconds
        :param end: the time of the stimulus h_offset in seconds
        :param ax: a plt axis
        :param fs: int, the sampling frequncy asociated with the data, used to plot with real time on the x axis
        :param ci: bool, wheter or not to plot athe bootstrap confidence itnernval
        :param y_offset: 'auto' or a number. when plotting multiple channels, the vertical h_offset between the PSTH of each
                          cell.
        :param plt_kws: adittional keyword parameters for pyplot.plot()
        :return: the ax used for the plotting
        '''

    # handles axis to plot on
    if ax == None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if times is not None:
        t = times
    else:
        t = np.arange(0, values.shape[1], values.shape[1])

    # gets the mean across repetitions i.e. dim 0
    psth = np.nanmean(values, axis=0)

    # gets the overall max spike rate to use as vertical end between channels
    if y_scaling != None:
        psth = psth * y_scaling

    if y_offset != None:
        psth += y_offset

    # Determine keyword arguments for pyplot.plot
    plt_kws = {} if plot_kws is None else plot_kws

    ax.plot(t, psth, **plt_kws)

    if ci == True:
        # todo inplement bootstarped confidence interval
        raise NotImplemented('implement it slacker!')
        ax.fill_between(t, conf_int[:, 0], conf_int[:, 1], color=color, alpha=0.2)

    # defiens the start and end of the sound
    if start != None:
        start = start
        ax.axvline(start, color='black')

    if end != None:
        end = end
        ax.axvline(end, color='black')

    ax.set_ylabel('spike rate (Hz)')

    return ax, fig, (np.min(psth), np.max(psth))


def _cint(times, values, confidence, ax=None, fillkwargs={}):
    """
    draws a confidence interval of the mean for the input array over the specified axis
    :param values: ndarray. 2dim R x T
    :param confidence: float. between 0 and 1
    :axis: int. default 1, it should be the repetions axis
    :param x: x axis values. it should be time
    :param ax: plt.ax.
    :param fillkwargs: dict. kwarguments for plt.fill_between
    :return: ax
    """
    #ToDo check that this is actually working and is not fishy
    if ax is None:
        fig, ax = plt.subplots()

    tails = (1 - confidence) / 2
    low = tails * 100
    high = (1 - tails) * 100
    lower, upper = np.percentile(values, [low, high], axis=0)

    ax.fill_between(times, lower, upper, **fillkwargs)

    return ax


def _neural_trajectory(matrix, dims=2, downsample=None, smoothing=0, rep_scat=True, rep_line=False,
                       mean_scat=False, mean_line=True):
    # TODO documentation
    # Base function for plotting a neuronal trajectory from a 3d matix with dimentions R x C x T where R is the repetitions,
    #     C is the cell and T is time. This matrix usually comes from "extract_epochs"

    # parses dimensions
    if dims == 3:  # default, first 3 dimentions
        matrix = matrix[:, 0:3, :]

    elif dims == 2:  # flat, two dimentions
        matrix = matrix[:, 0:2, :]

    elif isinstance(dims, list):  # specified channels
        if len(dims) > 3: raise ValueError("'dims' parameter cannot be more tha 3 and its {}".format(len(dims)))
        matrix = np.take(matrix, dims, axis=1)

    matrix_mean = np.expand_dims(np.mean(matrix, axis=0), axis=0)

    # downsamples across time by the factor specified
    if downsample != None:
        matrix = ssig.decimate(matrix, downsample, axis=2)

    # smooths the matrixes  with a gaussian filter along the time dimention
    matrix = sf.gaussian_filter(matrix, [0, 0, smoothing])
    matrix_mean = sf.gaussian_filter(matrix_mean, [0, 0, smoothing])

    # temporal axis (N) and colormap based on time.
    N = matrix.shape[2]
    cmap = plt.cm.jet(np.linspace(0, 1, N))

    ## 2D plotting
    if matrix.shape[1] == 2:
        fig, ax = plt.subplots()

        # plots the repetitions
        if rep_scat or rep_line:
            for rr in range(matrix.shape[0]):
                x = matrix[rr, 0, :]
                y = matrix[rr, 1, :]

                # scatter
                if rep_scat == True:
                    ax.scatter(x, y, c=cmap, alpha=0.1)

                # line
                if rep_line == True:
                    for ii in range(N - 1):
                        ax.plot(x[ii:ii + 2], y[ii:ii + 2], color=cmap[ii, :], alpha=0.1)

        # plots the mean
        x_mean = matrix_mean[0, 0, :]
        y_mean = matrix_mean[0, 1, :]

        # scatter
        if mean_scat == True:
            ax.scatter(x_mean, y_mean, c=cmap, alpha=0.1)

        # line
        if mean_line == True:
            for ii in range(N - 1):
                ax.plot(x_mean[ii:ii + 2], y_mean[ii:ii + 2], color=cmap[ii, :], alpha=1)

        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')


    ## 3D plotting
    elif matrix.shape[1] == 3:

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # plots the repettions
        if rep_scat or rep_line:
            for rr in range(matrix.shape[0]):
                x = matrix[rr, 0, :]
                y = matrix[rr, 1, :]
                z = matrix[rr, 2, :]

                # scatter
                if rep_scat == True:
                    ax.scatter(x, y, z, color=cmap, alpha=0.1)

                # line
                if rep_line == True:
                    for ii in range(N - 1):
                        ax.plot(x[ii:ii + 2], y[ii:ii + 2], z[ii:ii + 2], color=cmap[ii, :], alpha=0.1)

        # plot the mean
        x = matrix_mean[0, 0, :]
        y = matrix_mean[0, 1, :]
        z = matrix_mean[0, 2, :]

        # scatter
        if mean_scat == True:
            ax.scatter(x, y, z, color=cmap, alpha=0.5)
        # line
        if mean_line == True:
            for ii in range(N - 1):
                ax.plot(x[ii:ii + 2], y[ii:ii + 2], z[ii:ii + 2], color=cmap[ii, :], alpha=1)

        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
        ax.set_zlabel('component 3')

    return fig, ax


def _significance_bars(start_times, end_times, y_range='auto', ax=None, fill_kws=None):
    if ax == None:
        fig, ax = plt.subplots()
        y = (0, 1)
    else:
        fig = ax.figure

        if y_range == 'auto':
            y = ax.get_ylim()
        else:
            y = y_range

    # updates kwargs
    fill_kws = {} if fill_kws is None else fill_kws
    defaults = {'color': 'gray', 'alpha': 0.5}
    for key, arg in defaults.items(): fill_kws.setdefault(key, arg)

    # plots
    for start, end in zip(start_times, end_times):

        # skips if no significant bins/times i.e. an empty array
        if start.size != 0:
            # not empty
            ax.fill_betweenx(y, start, end, **fill_kws)

    return fig, ax


def dispersion(matrixes, smoothing=0, y_offset=0, ax=None, rep_line=None, mean_line=None, line_kws=None):
    # this is super temporary bad implementeation, again each subplot should be distintive of a single cell
    #  matrixes is a dict of matrix each of which has dimentions C x T (cells x Time). each different matrix
    #  corresponds to a different group of subepochs i.e. all conntexts for a given prb

    stimuli = list(matrixes.keys())

    # stacks all matrixes by a new axis represetning epch
    matrix = np.stack(matrixes.values(), axis=2)

    # this is such a cludge! better handling of channels and subplotting.
    fig, axes = subplots_sqr(matrix.shape[0])

    # iterates over subplots/cells

    for cell, ax in zip(range(matrix.shape[0]), axes):

        offset_counter = 0
        # iterates over stimuli
        for stim, stim_name in enumerate(stimuli):
            sliced = matrix[cell, :, stim] + (y_offset * offset_counter)
            ax.plot(sliced.T, alpha=0.2, label=stim_name)
            offset_counter += 1

        mean = np.nanmean(matrix[cell, :, :],
                          axis=1) * offset_counter  # scales by the total amoutn of different stimuli
        ax.plot(mean, color='black', label='mean')

        ax.set_title('channel {}'.format(cell))
        if cell == 0:
            ax.legend()

    return fig, axes


##### signal and recording wrapers

def signal_PSTH(signal, epoch_names='single', channels='all', psth_kws=None, plot_kws=None):
    '''
    plots PSTHs for a CPP sig. Generates a figure with m  axes where m is the eps ploted

    :param recording: a sig object with CPP formatted eps (see cpp_epochs)
    :param epoch_names: 'single', 'pair', regex, list.
                        'single' plots the PSTH of each CPP vocalization independent of its stim_num
                        'pair' plots idependent PSTHs for each contex prb pair
                        a regular expression, or a list of epoch names (str)
    :return: fig, axes
    '''

    # Determine keyword arguments for PSTH and plot
    psth_kws = {} if psth_kws is None else psth_kws
    plot_kws = {} if plot_kws is None else plot_kws

    # handles epochs and channel values/keywords
    epoch_names = _epoch_name_handler(signal, epoch_names)
    channels = _channel_handler(signal, channels)

    fig, axes = subplots_sqr(len(channels))

    signal = signal.rasterize()

    # extract the specified eps
    matrix_dict = signal.extract_epochs(epoch_names)

    first = True
    # iterates over each cell / subplot
    for chan, ax in zip(channels, axes):
        # iterates over each subepoch
        for cc, (epoch_key, epoch_matrix) in enumerate(matrix_dict.items()):
            color = 'C{}'.format(cc % 10)
            y_offset = len(epoch_names) * cc  # offsets subsequent psths

            values = epoch_matrix[:, chan, :]  # holds all Repetitions, for a given Channel, acrossTime
            times = np.arange(0, epoch_matrix.shape[2]) / signal.fs

            plot_kws.update({'color': color,
                             'label': epoch_key})

            _PSTH(times, values, ax=ax, y_offset=y_offset, **psth_kws, plot_kws=plot_kws)

        ax.set_title('{}: {}'.format(chan, signal.chans[chan]))
        if first:
            first = False
            ax.legend()
    fig.suptitle(signal.name)

    return fig, axes


def recording_PSTH(recording, epoch_names='single', signal_names='all', channels='all', psth_kws=None):
    '''
    plots PSTHs for a CPP recording. Generates a figure with m * n axes where m is the eps and n is the signals ploted

    :param recording: a recording object with CPP formated eps (see cpp_epochs)
    :param epoch_names: 'single', 'pair', regex, list.
                        'single' plots the PSTH of each CPP vocalization independent of its stim_num
                        'pair' plots idependent PSTHs for each contex prb pair
                        a regular expression, or a list of epoch names (str)
    :param signal_names: 'all', list of str. 'all' uses all eps. otherwise only the eps specified as a list
    :return: fig, axes
    '''

    # todo make this use the signal_PSTH instead., no need to  plot different signals in the same figure.

    if signal_names == 'all':
        signal_names = recording.signals.keys()

    # Determine keyword arguments for psth
    psth_kws = {} if psth_kws is None else psth_kws
    psth_kws.update({'channels': channels})

    epoch_names = _epoch_name_handler(recording, epoch_names)

    # creates a figure with apropiate number of row/cols of axes
    fig, axes = plt.subplots(len(signal_names), len(epoch_names))
    if len(signal_names) == 1 and len(epoch_names) == 1:  # if single ax
        axes = np.array(axes, ndmin=2)  # orders in a 2 dim array
    elif len(signal_names) == 1:  # if 1 dim array
        axes = np.expand_dims(axes, 0)  # adds one empty dimention
    elif len(epoch_names) == 1:  # if 1 dim array
        axes = np.expand_dims(axes, 1)  # adds one empty dimention

        # iterate over each sig in the recording
    for ss, sig_key in enumerate(signal_names):
        signal = recording[sig_key]
        signal = signal.rasterize()

        # extract the specified eps
        epochs_dic = signal.extract_epochs(epoch_names)

        for ee, (epoch_name, matrix) in enumerate(epochs_dic.items()):
            ax = axes[ss, ee]
            psth_kws.update({'ax': ax})
            ax = _PSTH(matrix, **psth_kws)
            ax.set_title("Signal: {}, Epoch: {}".format(sig_key, epoch_name))

    return fig, axes


def signal_trajectory(signal, dims=2, epoch_names='single', _trajectory_kws=None):
    # handles _neural_trajectory kwargs
    _trajectory_kws = {} if _trajectory_kws == None else _trajectory_kws
    _trajectory_kws.update({'dims': dims})

    sig_name = signal.name
    sufix = sig_name.split('_')[-1]
    if sufix != 'PCs':
        warnings.warn('plotting neural trajectory over raw dimentions, use principal components instead')

    # handles epochs names
    epoch_names = _epoch_name_handler(signal, epoch_names)

    matrix_dict = signal.rasterize().extract_epochs(epoch_names)

    for epoch_key, matrix in matrix_dict.items():
        fig, ax = _neural_trajectory(matrix, **_trajectory_kws)
        ax.set_title(epoch_key)

    return fig, ax  # todo, should i return something better??


def recording_trajectory(recording, dims=2, epoch_names='single', signal_names='PCA', _trajectory_kws=None):
    if signal_names == 'all':
        signal_names = recording.signals.keys()
    elif signal_names == 'PCA':
        signal_names = [name for name in recording.signals.keys() if name.split('_')[-1] == 'PCs']
    else:
        signal_names = [signal_names]

    figs = list()

    for ii, sig_name in enumerate(signal_names):
        signal = recording[sig_name]
        fig, ax = signal_trajectory(signal, dims=dims, epoch_names=epoch_names, _trajectory_kws=_trajectory_kws)
        fig.suptitle('sig_name')
        figs.append(fig)
        return figs


def signal_raster(signal, epoch_names='single', channels='all', scatter_kws=None):
    # todo documentation
    epoch_names = _epoch_name_handler(signal, epoch_names)
    matrixes = signal.rasterize().extract_epochs(epoch_names)

    # handles scatter_kws
    scatter_kws = {} if scatter_kws is None else scatter_kws

    channels = _channel_handler(signal, channels=channels)

    # defines the number and distribution of subplots in the figure
    fig, axes = subplots_sqr(len(channels))

    # iterates over each cell... plot
    first = True
    for chan, ax in zip(channels, axes):

        for cc, (epoch, matrix) in enumerate(matrixes.items()):
            color = 'C{}'.format(cc % 10)
            y_offset = matrix.shape[0] * cc  # offsets subsequent rasters by the number of repetitiond

            values = matrix[:, chan, :]  # holds all Repetitions, for a given Channel, acrossTime
            times = np.arange(0, matrix.shape[2]) / signal.fs

            scatter_kws.update({'color': color,
                                'label': epoch})

            _raster(times, values, y_offset=y_offset,
                    ax=ax, scatter_kws=scatter_kws)

        ax.set_title('{}: {}'.format(chan, signal.chans[chan]))
        if first:
            first = False
            ax.legend()
    fig.suptitle(signal.name)

    return fig, matrixes


def hybrid(signal, epoch_names='single', channels='all', start=None, end=None,
           significance=None, raster_fs=None, psth_fs=None, sign_fs=None, sign_kws=None,
           scatter_kws=None, plot_kws=None, sub_types=(True, True, True), time_strech=None, time_offset=None,
           legend=True, labels=None, colors=None, axes=None):
    '''
    Todo clean documentation
    given a signal, creates a figure where each subplots correspond to a cell in that signal, where each subplot contains
    a raster, psth and indications of siginificant difference over time. different epochs are represented by different
    colors within each subplot
    :param signal: A nems signals with epochs formated under for stim_num prb pairs, see cpp_epochs
    :param epoch_names: str, regexp or keyword(str) to select what epochs to extract and use to plot. the keyword 'single'
    indicated use of a the ressponse of to a sound regardles of the preceding stim_num. Keyword 'pair' independently plots
    all possible different stim_num-prb pairs
    :param channels: int, str, [int, ...], [str, ...], Keyword(str). Selects what cells in the population to plot,
    and therefore controlls the number of subplots. specific cells can be specified by their index or name, and multiple
    cells can be specified as list of the previous. Keyword 'all' uses all cells available in the signal
    :param start: float, in seconds. Specify where to draw a vertical line to indicate start of the stimulus
    :param end: float, in seconds. Specify where to draw a vertical line to indicate end of the stimulus
    :param significance: array of shape Cell x time. both dimensions must be consistent with the dimensions of the the
    number of cells ploted and the time of the raster/PSTH bein ploted, in units of time bins.
    :param raster_fs: float, None. Samplig frequency for the raster. None specifies native resolution(recomended)
    :param psth_fs: float, None. Samplig frequency for the raster. None specifies native resolution, downsampling recomended
    :param sign_fs: float, None. Samplig frequency for the raster. None specifies native resolution(recomended)
    :param sign_kws: aditional kwargs to pass into _sig_bin_to_time
    :param scatter_kws: aditional kwargs to pass into _raster
    :param plot_kws: aditional kwargs to pass into _PSTH
    :param sub_types: [bool, bool, bool], defines whether to use or not any of the plot subtypes: [raster, psth, significance]
    :param axes: list of plt axes

    :return:
    '''
    if not isinstance(signal, PointProcess):
        warnings.warn('signal is not a PointProcess, resampling not implemented')
        can_resample = False
    else: can_resample = True

    # formats epoch naems
    epoch_names = _epoch_name_handler(signal, epoch_names)

    # formats fs paramter
    if can_resample is True:
        raster_fs = _fs_handler(signal, raster_fs)
        psth_fs = _fs_handler(signal, psth_fs)
        sign_fs = _fs_handler(signal, sign_fs)
        rast_matrices = signal._modified_copy(signal._data, fs=raster_fs).rasterize().extract_epochs(epoch_names)
        psth_matrices = signal._modified_copy(signal._data, fs=psth_fs).rasterize().extract_epochs(epoch_names)
    elif can_resample is False:
        # uses native samplitng frequecy
        raster_fs = signal.fs
        psth_fs = signal.fs
        sign_fs = signal.fs
        rast_matrices = signal.rasterize().extract_epochs(epoch_names)
        psth_matrices = signal.rasterize().extract_epochs(epoch_names)
    # sets the values in Hz
    psth_matrices = {key: val * psth_fs for key, val in psth_matrices.items()}

    # cuts the specific strech of time to be ploted
    if time_strech != None:
        time_strech = np.array(time_strech)
        psth_strech = np.floor(time_strech*psth_fs).astype(int)
        rast_strech = np.floor(time_strech*raster_fs).astype(int)
        psth_matrices = {key: mat[:,:, psth_strech[0]:psth_strech[1]] for key, mat in psth_matrices.items()}
        rast_matrices = {key: mat[:, :, rast_strech[0]:rast_strech[1]] for key, mat in rast_matrices.items()}
    # offsets the x axis time
    if time_offset == None: time_offset = 0

    # preprocesing of significance array
    if isinstance(significance, np.ndarray):
        # handles significance keywords
        sign_kws = {} if sign_kws is None else sign_kws
        # some preprocesing of the significance matrix
        defaults = {'window': 1}
        for key, val in defaults.items(): sign_kws.setdefault(key, val)

        start_times, end_times = _sig_bin_to_time(significance, fs=sign_fs, **sign_kws)

    # handles scatter and psth keywords
    scatter_kws = {} if scatter_kws is None else scatter_kws
    plot_kws = {} if plot_kws is None else plot_kws

    channels = _channel_handler(signal, channels=channels)

    # defines the number and distribution of subplots in the figure
    if axes is None:
        fig, axes = subplots_sqr(len(channels))
    elif axes is not None:
        if len(axes)<len(channels): raise ValueError(f'{len(channels)} axes required but {len(axes)} pased')
        fig = axes[0].get_figure()

    # calculates the vertical_range of the significance areas based on the number of stimuli and repetitionse per stimuli
    # in the second value, the first factor is the number of stimuli, the second is the number of repetitions
    vertical_range = [-0.5, len(psth_matrices) * psth_matrices[epoch_names[0]].shape[0] + 0.5]

    # iterates over each cell... plot
    for cc, (chan, ax) in enumerate(zip(channels, axes)):
        # saves y ofsets for proper y axis labeling
        y_ticks = list()
        y_ticklabels = list()

        y_offset = np.max(np.nanmean(np.stack(psth_matrices.values(), axis=3)[:,chan,:,:], axis=0)) * 1.1
        for ss, (epoch_key, epoch_matrix) in enumerate(psth_matrices.items()):
            color = colors[ss] if colors is not None else 'C{}'.format(ss % 10)
            y_off = y_offset * ss  # offsets subsequent rasters by the number of repetitions

            # set lables to default of personalized
            if labels is not None:
                if len(labels) != len(psth_matrices):
                    raise ValueError(f"{len(labels)} labels were given, but there are {len(psth_matrices)} different epochs")
                label = labels[ss]
            else:
                label = epoch_key

            # PSTH
            if sub_types[1] is True:
                # values for psth
                values = epoch_matrix[:, chan, :]  # holds all Repetitions, for a given Channel, acrossTime
                times = (np.arange(0, epoch_matrix.shape[2]) / psth_fs) + time_offset
                plot_kws.update({'color': color,
                                 'label': label})
                _, _, y_range = _PSTH(times, values, start=start, end=end, ax=ax, ci=False,
                                      y_offset=y_off, plot_kws=plot_kws)

                y_ticks.extend(y_range)
                y_ticklabels.extend([yy - y_off for yy in y_range])

            # Raster
            if sub_types[0] is True:
                # values for raster plot
                rast_epoch_matrix = rast_matrices[epoch_key]
                rast_values = rast_epoch_matrix[:, chan, :]  # holds all Repetitions, for a given Channel, acrossTime
                rast_times = (np.arange(0, rast_epoch_matrix.shape[2]) / raster_fs) + time_offset
                scatter_kws.update({'color': color})
                if not y_range:
                    y_range = None
                try:
                    # this will break if the raster is empty
                    _raster(rast_times, rast_values, y_offset=y_off, y_range=y_range,
                            ax=ax, scatter_kws=scatter_kws)
                except:
                    continue


        # set the y ticks and labels so they match the bottom and top values of PSTHs
        if sub_types[1] is True:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(['{:.0f}'.format(ylab) for ylab in y_ticklabels])


        # dispersion significance plotting
        if isinstance(significance, np.ndarray) and sub_types[2] is True:
            # draws the dispersion significance across time
            _significance_bars(start_times[cc], end_times[cc], y_range=vertical_range, ax=ax)

        ax.set_title('{}: {}'.format(chan, signal.chans[chan]))
    else:
        #ax.legend(bbox_to_anchor=(2, 1))
        if legend == True:
            ax.legend()
        elif legend == False:
            pass
        else:
            raise ValueError('legend must be boolean')
        pass

    fig.suptitle(signal.name)

    return fig, axes

##### Specializede plots


def plot_dist_with_CI(real, bootstrapped, labels, colors, smp_start, smp_end, smp_line, fs, ax=None, show_labels=False):

    # todo make a function that generates distributions not centered around zero
    '''
    plot the output of signal_single_trial_dispersion_pooled_shuffled, shows the distance over time with the
    confidence interval, between the specified start and end times
    :param real: vector, time series of distance over time
    :param bootstrapped: list of 2d array of time series vectors, with dimensions Repetition x Time, Time must be equal to 'real'
    :labels: list of string, labels for each of the bootstrapped distributions
    :colors: list of matplotlib colors for each of the bootstrapped distributions
    :param smp_start: start time in samples to plot from
    :param smp_end: end time in samples to plot until
    :smp_line: vertical line position in samples, or None
    :param fs: sampling frequency in Hz
    :show_labels: boolean. set some defaul axis labels
    :return: figure and ax handles
    '''

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    t1 = (smp_start / fs)
    t2 = (smp_end / fs)
    t = np.linspace(t1, t2, smp_end-smp_start)

    line = real[smp_start:smp_end]
    ax.plot(t, line, color='black')

    colors = colors if colors is not None else [f'C{x}' for x in range(len(bootstrapped))]

    for ii, (label, array, color) in enumerate(zip(labels, bootstrapped, colors)):
        shade = array[:, smp_start:smp_end]
        shade = np.mean(shade, axis=0) + np.std(shade, axis=0) * 2
        ax.fill_between(t, -shade, shade, alpha=0.5, color=color,label=label)

    if smp_line is not None:
        ax.axvline(smp_line/fs, color='black', linestyle='--')

    if show_labels:
        ax.set_xlabel('time (s)')
        ax.set_ylabel('euclidean distance')
        ax.tick_params(axis='both', which='major')

    return fig, ax


def exp_decay(times, values, ax=None, label=True, **pltkwargs):
    '''
    plots an exponential decaye curve fited on times and values
    :param times:
    :param values:
    :param ax:
    :param label:
    :param pltkwargs:
    :return:
    '''
    defaults = {'color': 'gray', 'linestyle': '--'}
    for key, arg in defaults.items(): pltkwargs.setdefault(key, arg)

    if ax == None:
        fig, ax = plt.subplots()
    else:
        ax = ax
        fig = ax.get_figure()

    try:
        popt, pcov, r2 = fts.exp_decay(times, values)
    except:
        print('failed to fit exp decay')
        return fig, ax, None, None

    if label == True:
        # scientific notation if number is too big
        to_display = [popt[0], -1 / popt[1]]
        formated = list()
        for vv, value in enumerate(to_display):
            if value >= 1000:
                formated.append('{:.2e}'.format(value))
            else:
                formated.append('{:.2f}'.format(value))

        label = 'r0={}\nt={}'.format(formated[0], formated[1])
    elif label == False:
        label = None
    else:
        pass

    ax.plot(times, fts._exp(times, *popt), label=label, **pltkwargs)

    return fig, ax, popt, pcov, r2, fts._exp(times, *popt)


def lin_reg(x, y, ax=None, label=True, **pltkwargs):
    defaults = {'color': 'gray', 'linestyle': '--', 'marker':None}
    for key, arg in defaults.items(): pltkwargs.setdefault(key, arg)


    reg = linregress(x, y)
    m, b, r, _, _ = reg

    if ax == None:
        fig, ax = plt.subplots()
    else:
        ax = ax
        fig = ax.get_figure()

    if label == True:
        # scientific notation if number is too big
        label = 'm={:.2f}, b={:.2f}, r={:.2f}'.format(m, b, r)

    ax.plot(x, m*x+ b, label=label, **pltkwargs)

    return fig, ax, reg

def unit_line(ax, square_shape=False, **pltkwargs):

    defaults = {'linestyle': '--', 'color': 'gray'}
    for key, arg in defaults.items(): pltkwargs.setdefault(key, arg)

    plot_lim = np.stack([ax.get_xlim(), ax.get_ylim()], axis=0)
    if square_shape is False:
        # finds the minimum common range between x and y axis
        range = np.min(plot_lim, axis=0)

    if square_shape is True:
        # finds the max comon range across x and y axis
        range = np.max(plot_lim, axis=0)

    ax.plot(range, range, **pltkwargs)

    return ax

def model_progression(x, y, data, mean, ax, order, palette, collapse_by=None, swarm=True, **pltKwargs):

    # ToDo rewite for publications??
    '''
    for every cell in the DF plots a point in each column corresponding to a model, and draws a line conecting the points
    this shows the progression of the value across models in a cell dependent manner
    :param x: str, column name corresponding to the categorical variable
    :param y: str, column name corresponding to the values to plot
    :param data: DF in tidy format, with a column corresponding to a categorical variable, and another to numerical values
    :param mean: bool, whether to plot the mean of the population progression
    :param ax: a matplotlib ax object
    :param order: list, tuple. the order in which the values of the categorical variables are plotted in the x axis
    :param palette: list, the colors in which the values at each model column are ploted
    :pltKwargs: aditional arguments to be passed to plt.plot()
    :return: the ax object containing the plot.
    '''
    # hold names of the to be columns
    col_names = set(data[x].unique())
    if col_names != set(order):
        raise ValueError('parameter Order specified category levels not in x')

    # starts by pivoting the dataframe into wideformat
    # wide = make_tidy(data, pivot_by=x, more_parms=None, values=y)
    wide = None
    # hold only the important colums in the right order
    wide = wide.loc[:, order]

    # gets the data as an array to be ploted by plt.plot
    toplot_arr = wide.values.T

    if collapse_by is None:
        pass

    elif 0<= collapse_by < len(order)-1 and isinstance(collapse_by, int):
        # chooses a level of the categorical variable, substracts the values of such level from all other levels
        #  to only show relative change, offsets to the mean of the values of the selected level
        normalizer = toplot_arr[collapse_by,:]
        toplot_arr = toplot_arr - normalizer[:] + np.mean(normalizer)

    else: raise ValueError('order should be a integer corresponding to the column number to collapse by')

    # plots individual lines
    ax.plot(toplot_arr,color='gray', alpha=0.4)
    if mean == True:
        ax.plot(np.mean(toplot_arr, axis=1), color='black')


    # plot individual scatters for each column
    for cc in range(len(order)):
        yy = toplot_arr[cc, :]
        xx = np.zeros(shape=yy.shape) + cc
        ax.scatter(xx, yy, color=palette[cc], edgecolor='black', linewidth='1',)

    # formats the x axis to be like a categorical variable
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(list(order))

    return ax


##### Other functions

def savefig(fig, root, name, type='png'):
    root = pl.Path(config['paths']['figures']) / f'{root}'
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)

    if type == 'png':
        png = root.joinpath(name).with_suffix('.png')
        fig.savefig(png, transparent=False, dpi=100)
    elif type == 'svg':
        svg = root.joinpath(name).with_suffix('.svg')
        fig.savefig(svg, transparent=True)
    else:
        raise ValueError(f"type must be 'png' or 'svg' but {type} was given")

    return None