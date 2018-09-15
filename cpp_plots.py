import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import nems.epoch as nep
import warnings
import scipy.ndimage.filters as sf
from nems.signal import SignalBase
import math


# todo redo the distribution of axes in figures for psth, there is not purpose in plotig multiple signals as different
# rows, rather the subplotse should correspond to individual cells, and elements within each axis should correspond to
# responses to different stimuli ...

### helper functions

def _epoch_name_handler(rec_or_sig, epoch_names):
    '''
    helper function to transform heterogeneous inputs of epochs names (epoch names, list of epochs names, keywords) into
    the corresponding list of epoch names.
    :param rec_or_sig: nems recording of signal object
    :param epoch_names: epoch name (str), regexp, list of epoch names, 'signle', 'pair'. keywords 'single' and 'pair'
    correspond to all signle vocalization, and pair of context probe vocalizations.
    :return: a list with the apropiate epoch names as found in signal.epoch.name
    '''
    if epoch_names == 'single':  # get eps matching 'voc_x' where x is a positive integer
        reg_ex = r'\Avoc_\d'
        epoch_names = nep.epoch_names_matching(rec_or_sig.epochs, (reg_ex))
    elif epoch_names == 'pair':  # get eps matching 'Cx_Py' where x and y are positive integers
        reg_ex = r'\AC\d_P\d'
        epoch_names = nep.epoch_names_matching(rec_or_sig.epochs, (reg_ex))
    elif isinstance(epoch_names, str):  # get eps matching the specified regexp
        reg_ex = epoch_names
        epoch_names = nep.epoch_names_matching(rec_or_sig.epochs, (reg_ex))
    elif isinstance(epoch_names, list):  # uses epoch_names as a list of epoch names.
        ep_intersection = set(epoch_names).intersection(set(rec_or_sig.epochs.name.unique()))
        if len(ep_intersection) == 0:
            raise AttributeError("specified eps are not contained in sig")
        pass

    if len(epoch_names) == 0:
        raise AttributeError("no eps match regex '{}'".format(reg_ex))

    return epoch_names


def _channel_handler(mat_or_sig, channels):
    '''
    Helper function to handle heterogeneous inputs to channel parameter (index, list of indexes or cell names, keywords)
    and returns an homogeneous list of indexes.
    :param mat_or_sig: 3d matrix with shape R x C x T (rep, chan, time), or signal object.
    :param channels: Channel index (int) or list of index, cell name (str) or list of names, 'all'. keyword 'all' includes
    all channels/cells in the signal/matrix.
    :return: list of channels indexes.
    '''
    # chekcs the object type of the parameters
    if isinstance(mat_or_sig, np.ndarray):
        max_chan = mat_or_sig.shape[1]

    elif isinstance(mat_or_sig, SignalBase):
        is_sig = True
        max_chan = mat_or_sig.nchans


    # returns a different list of channles depending on the keywords or channels specified. add keywords here!
    if channels == 'all':
        plot_chans = range(max_chan)
    elif isinstance(channels, int):
        if channels >= max_chan:
            raise ValueError('recording only has {} channels, but channels value {} was given'.
                             format(max_chan, channels))
        plot_chans = [channels]
    elif isinstance(channels, list) :
        item = channels[0]
        # list of indexes
        if isinstance(item, int):
            for chan in channels:
                if chan > max_chan:
                    raise ValueError('recording only has {} channels, but channels value {} was given'.
                                     format(max_chan, channels))
            plot_chans = channels
        # list of cell names
        elif isinstance(item, str):
            if is_sig != True:
                raise ValueError('can only use cell names when indexing from a signal object')
            plot_chans = [mat_or_sig.chans.index(channels)]

    elif isinstance(channels, str):
        # accepts the name of the unit as found in cellDB
        if is_sig != True:
            raise ValueError('can only use cell names when indexing from a signal object')
        plot_chans = [mat_or_sig.chans.index(channels)]

    return plot_chans


def _subplot_handler(epoch_names, channels):
    ax_num = len(channels)

    rows = int(np.ceil(math.sqrt(ax_num)))
    cols = int(np.floor(math.sqrt(ax_num)))

    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, squeeze=False)

    axes = np.ravel(axes)

    return fig, axes


### base plotting functions


def _raster(times, values, y_offset=None, ax=None, scatter_kws=None):
    '''
    Plots a raster with one line for each pair of
    time and value vectors.
    Lines will be auto-colored according to matplotlib defaults.

    times : Array with shape T where T is time bins in seconds
    values : Array with shape R x T : where R is repetitions, and T is time. the dimention of T must agree with that of
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

    if y_offset != None:
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


def _PSTH2(times, values, start=None, end=None, ax=None, ci=False, y_offset=None, y_scaling=None, plot_kws=None):
    '''
        Base function for plotting a PSTH from a 2d matix with dimentions R x T where R is the repetitions
        and T is time. This matrix usually comes from "extract_epochs" afeter selecting a specific channel/cell
        :times: Array with shape T where T is time bins in seconds
        :values: Array with shape R x T : where R is repetitions, and T is time. the dimention of T must agree with that of T in times.
        :param start: the time of the sound onset in seconds
        :param end: the time of the stimulus offset in seconds
        :param ax: a plt axis
        :param fs: int, the sampling frequncy asociated with the data, used to plot with real time on the x axis
        :param ci: bool, wheter or not to plot athe bootstrap confidence itnernval
        :param y_offset: 'auto' or a number. when plotting multiple channels, the vertical offset between the PSTH of each
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

    return ax, fig


def _neural_trajectory(matrix, dims=2, smoothing=0, rep_scat=True, rep_line=False,
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


##### signal and recording wrapers


def signal_PSTH(signal, epoch_names='single', channels='all', psth_kws=None, plot_kws=None):
    '''
    plots PSTHs for a CPP sig. Generates a figure with m  axes where m is the eps ploted

    :param recording: a sig object with CPP formatted eps (see cpp_epochs)
    :param epoch_names: 'single', 'pair', regex, list.
                        'single' plots the PSTH of each CPP vocalization independent of its context
                        'pair' plots idependent PSTHs for each contex probe pair
                        a regular expression, or a list of epoch names (str)
    :return: fig, axes
    '''

    # Determine keyword arguments for PSTH and plot
    psth_kws = {} if psth_kws is None else psth_kws
    plot_kws = {} if plot_kws is None else plot_kws

    # handles epochs and channel values/keywords
    epoch_names = _epoch_name_handler(signal, epoch_names)
    channels = _channel_handler(signal, channels)

    fig, axes = _subplot_handler(epoch_names, channels)

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

            _PSTH2(times, values, ax=ax, y_offset=y_offset, **psth_kws, plot_kws=plot_kws)

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
                        'single' plots the PSTH of each CPP vocalization independent of its context
                        'pair' plots idependent PSTHs for each contex probe pair
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
    fig, axes = _subplot_handler(epoch_names, channels)

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


def hybrid(signal, epoch_names='single', channels='all', start=None, end=None, scatter_kws=None, plot_kws=None):
    epoch_names = _epoch_name_handler(signal, epoch_names)
    matrixes = signal.rasterize().extract_epochs(epoch_names)

    # handles scatter_kws
    scatter_kws = {} if scatter_kws is None else scatter_kws
    plot_kws = {} if plot_kws is None else plot_kws

    channels = _channel_handler(signal, channels=channels)

    # defines the number and distribution of subplots in the figure
    fig, axes = _subplot_handler(epoch_names, channels)

    # scales the PSTH to entirely overlap with the raster
    max_val = np.max([np.max(np.mean(epoch_matrix, axis=0)) for epoch_matrix in matrixes.values()])
    max_Reps = np.max([epoch_matrix.shape[0] for epoch_matrix in matrixes.values()])
    y_scaling = max_Reps / max_val  # normalizes by max val and scales to max reps

    # iterates over each cell... plot
    first = True
    for chan, ax in zip(channels, axes):

        for cc, (epoch_key, epoch_matrix) in enumerate(matrixes.items()):
            color = 'C{}'.format(cc % 10)
            y_offset = epoch_matrix.shape[0] * cc  # offsets subsequent rasters by the number of repetitions

            values = epoch_matrix[:, chan, :]  # holds all Repetitions, for a given Channel, acrossTime
            times = np.arange(0, epoch_matrix.shape[2]) / signal.fs

            scatter_kws.update({'color': color})

            _raster(times, values, y_offset=y_offset,
                    ax=ax, scatter_kws=scatter_kws)

            plot_kws.update({'color': color,
                             'label': epoch_key})

            _PSTH2(times, values, start=start, end=end, ax=ax, ci=False, y_offset=y_offset, y_scaling=y_scaling,
                   plot_kws=plot_kws)

        ax.set_title('{}: {}'.format(chan, signal.chans[chan]))
        if first:
            first = False
            ax.legend()
    fig.suptitle(signal.name)

    return fig, matrixes


# do deprecate

def _PSTH(matrix, start=None, end=None, ax=None, fs=None, ci=False, y_offset='auto', channels='all', plt_kws=None):
    '''
    Base function for plotting a PSTH from a 3d matix with dimentions R x C x T where R is the repetitions,
    C is the cell and T is time. This matrix usually comes from "extract_epochs"
    :param matrix: a 3d numpy array
    :param start: the time of the sound onset in seconds
    :param end: the time of the stimulus offset in seconds
    :param ax: a plt axis
    :param fs: int, the sampling frequncy asociated with the data, used to plot with real time on the x axis
    :param ci: bool, wheter or not to plot athe bootstrap confidence itnernval
    :param y_offset: 'auto' or a number. when plotting multiple channels, the vertical offset between the PSTH of each
                      cell.
    :param plt_kws: adittional keyword parameters for pyplot.plot()
    :return: the ax used for the plotting
    '''

    # the dimentions of the matrix are repetitions, channels, and time in that order
    # defines in what axis to plot
    if ax == None:
        # creates a figure and plot
        fig, ax = plt.subplots()

    # defines wheter to use sampling frequency (and therefore time) in the x axis
    if isinstance(fs, int):
        period = 1 / fs

    elif fs == None:
        period = matrix.shape[2]  # uses time bins instead of time

    t = np.arange(0, matrix.shape[2] * period, period)

    # gets the mean across repetitions
    psth = np.nanmean(matrix, axis=0)  # psth has dimentions: Channels x time

    # gets the overall max spike rate to use as vertical end between channels
    if y_offset == 'auto':
        y_offset = np.max(psth)

    # Determine keyword arguments for pyplot.plot
    plt_kws = {} if plt_kws is None else plt_kws

    plot_chans = _channel_handler(matrix, channels)

    # iterates over every cell
    for channel in plot_chans:
        # offsets each channle for better readability
        toplot = psth[channel, :] - y_offset * channel
        ax.plot(t, toplot, **plt_kws)

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

    return ax
