import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nems.signal as signal
import seaborn as sns
import nems.epoch as nep


def _PSTH(matrix, onset=None, offset=None, ax=None, fs=None, ci=False, y_offset='auto', plt_kws=None):

    '''
    Base function for plotting a PSTH from a 3d matix with dimentions R x C x T where R is the repetitions,
    C is the channel and T is time. This matrix usually comes from "extract_epochs"
    :param matrix: a 3d numpy array
    :param onset: the time of the sound onset in seconds
    :param offset: the time of the stimulus offset in seconds
    :param ax: a plt axis
    :param fs: int, the sampling frequncy asociated with the data, used to plot with real time on the x axis
    :param ci: bool, wheter or not to plot athe bootstrap confidence itnernval
    :param y_offset: 'auto' or a number. when plotting multiple channels, the vertical offset between the PSTH of each
                      channel.
    :param plt_kws: adittional keyword parameters for pyplot.plot()
    :return: the ax used for the plotting
    '''

    # the dimentions of the matrix are repetitions, channels, and time in that order
    # defiens in what axis to plot
    if ax == None:
        # creates a figure and plot
        fig, ax = plt.subplots()

    # defines wheter to use sampling frequency (and therefore time) in the x axis
    if isinstance(fs, int):
        period = 1/fs

    elif fs == None:
        period = matrix.shape[2] # uses time bins instead of time

    t = np.arange(0, matrix.shape[2] * period, period)


    # gets the mean across repetitions
    psth = np.nanmean(matrix, axis=0) # psth has dimentions: Channels x time

    # gets the overall max spike rate to use as vertical offset between channels
    if y_offset == 'auto':
        y_offset =  np.max(psth)

    # Determine keyword arguments for the facets
    plt_kws = {} if plt_kws is None else plt_kws


    # iterates over every channel
    for channel in range(psth.shape[0]):
        # offsets each channle for better readability
        toplot = psth[channel, :] - y_offset * channel
        ax.plot(t, toplot, **plt_kws)


    if ci == True:
        # todo inplement bootstarped confidence interval
        raise NotImplemented('implement it slacker!')
        ax.fill_between(t, conf_int[:, 0], conf_int[:, 1], color=color, alpha=0.2)

    # defiens the onset and offset of the sound
    if onset != None:
        onset = onset * period
        ax.axvline(onset, color='black')

    if offset != None:
        offset = offset * period
        ax.axvline(offset, color='black')


    ax.set_ylabel('spike rate (Hz)')
    # todo handle legend?
    #ax.legend(loc='upper left', fontsize='xx-small')

    return ax


def epochs_PSTH(recording, epoch_names='single', signal_names ='all'):
    '''
    plots PSTHs for a CPP recording. Generates a figure with m * n axes where m is the epochs and n is the signals ploted

    :param recording: a recording object with CPP formated epochs (see cpp_epochs)
    :param epoch_names: 'single', 'pair', regex, list.
                        'single' plots the PSTH of each CPP vocalization independent of its context
                        'pair' plots idependent PSTHs for each contex probe pair
                        a regular expression, or a list of epoch names (str)
    :param signal_names: 'all', list of str. 'all' uses all epochs. otherwise only the epochs specified as a list
    :return: fig, axes
    '''

    if signal_names == 'all':
        signal_names = recording.signals.keys()

    # defines general versions of
    if epoch_names == 'single':
        reg_ex = r'\Avoc_\d'
        epoch_names = nep.epoch_names_matching(recording.epochs, (reg_ex))
    elif epoch_names == 'pair':
        reg_ex = r'\AC\d_P\d'
        epoch_names = nep.epoch_names_matching(recording.epochs, (reg_ex))
    elif isinstance(epoch_names, str):
        reg_ex = epoch_names
        epoch_names = nep.epoch_names_matching(recording.epochs, (reg_ex))
    elif isinstance(epoch_names, list):
        ep_intersection = set(epoch_names).intersection(set(recording.epochs.name.unique()))
        if len(ep_intersection) == 0:
            raise AttributeError("specified epochs are not contained in recording")
        pass

    if len(epoch_names) == 0:
        raise AttributeError("no epochs match regex '{}'".format(reg_ex))

    # creates a figure with apropiate number of row/cols of axes
    fig, axes = plt.subplots(len(signal_names), len(epoch_names))
    if len(signal_names) == 1 and len(epoch_names) == 1:   # if single ax
        axes = np.array(axes, ndmin=2)                 # orders in a 2 dim array
    elif len(signal_names) ==1:                        # if 1 dim array
        axes = np.expand_dims(axes, 0)                 # adds one empty dimention
    elif len(epoch_names) == 1:                        # if 1 dim array
       axes = np.expand_dims(axes, 1)                  # adds one empty dimention

        # iterate over each signal in the recording
    for ss, sig_key in enumerate(signal_names):
        signal = recording[sig_key]
        signal = signal.rasterize()

        # extract the specified epochs
        epochs_dic = signal.extract_epochs(epoch_names)

        for ee, (epoch_name, matrix) in enumerate(epochs_dic.items()):
            ax = axes[ss, ee]

            ax = _PSTH(matrix, ax=ax)
            ax.set_title("Signal: {}, Epoch: {}".format(sig_key, epoch_name))

    return fig, axes




# thise is just for reference
def psth(ctx, sub_epoch=None, super_epoch=None):
        # todo pull the fs from somewhere
        fs = 100
        period = 1 / fs

        meta = ctx['modelspecs'][0][0]['meta']

        # organizes data

        resp_pred = dict.fromkeys(['resp', 'pred'])

        # color represents sound frequency
        colors = ['green', 'red']
        frequencies = ['f1', 'f2']

        # linestyle indicates preentation rate
        linestyles = ['-', ':']
        rates = ['std', 'dev']

        # pull the signals from the validation recording in ctx
        for key in resp_pred.keys():
            signal = ctx['val'][0][key]

            # folds by oddball epochs
            folded_sig = of.extract_signal_oddball_epochs(signal, sub_epoch, super_epoch)

            resp_pred[key] = folded_sig

        # calculate confidence intervals, organizes in a dictionary of equal structure as the matrix
        conf_dict = {outerkey: {innerkey:
                                    np.asarray([bts.ci(innerval[:, 0, tt], np.mean, n_samples=100, method='pi')
                                                for tt in range(innerval.shape[2])])
                                for innerkey, innerval in outerval.items()} for outerkey, outerval in resp_pred.items()}

        fig, axes = plt.subplots(1, 2, sharey=True)

        axes = np.ravel(axes)

        for ax, RP in zip(axes, resp_pred.keys()):

            for color, freq in zip(colors, frequencies):

                for linestyle, rate in zip(linestyles, rates):
                    outerkey = RP
                    innerkey = freq + '_' + rate

                    matrix = resp_pred[outerkey][innerkey]
                    psth = np.nanmean(matrix, axis=0).squeeze()
                    conf_int = conf_dict[outerkey][innerkey]
                    onset = (psth.shape[0] / 3) * period
                    offset = (psth.shape[0] * 2 / 3) * period

                    t = np.arange(0, psth.shape[0] * period, period)

                    ax.plot(t, psth, color=color, linestyle=linestyle, label=innerkey)
                    ax.fill_between(t, conf_int[:, 0], conf_int[:, 1], color=color, alpha=0.2)

                    ax.axvline(onset, color='black')
                    ax.axvline(offset, color='black')

                    ax.set_ylabel('spike rate (Hz)')
                    ax.legend(loc='upper left', fontsize='xx-small')

            ax.set_title(outerkey)

        fig.suptitle('{} {}'.format(meta['cellid'], meta['modelname']))

        return fig, axes
