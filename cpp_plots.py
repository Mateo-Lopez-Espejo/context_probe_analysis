import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nems.signal as signal
import seaborn as sns


def _PSTH(matrix, onset=None, offset=None, ax=None, fs=None, ci=False, y_offset='auto', plt_kws=None):

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
        toplot = psth[channel, :] + y_offset * channel
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



def ndim_resp_PSTH(rec, signal='all', dims='all'):
    '''
    genenates a PSTH for each inncreased
    :param rec:
    :param signal:
    :param dims:
    :return:
    '''
    return None




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
