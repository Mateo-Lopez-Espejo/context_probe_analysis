import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
from scipy.stats import gaussian_kde as gkde

from dPCA import dPCA

from cpp_parameter_handlers import _channel_handler

import cpn_triplets as tp

def format_raster(raster):
    '''
    formats a CPP/CPN raster into the trial and mean arrays required for dPCA
    :param raster: np.array. with dimensions Context x Probe x Trial x Neuron x Time
    :return:
    '''
    # reorders dimentions from Context x Probe x Trial x Neuron x Time  to  Trial x Neuron x Context x Probe x Time
    trialR = raster.transpose([2, 3, 0, 1, 4])
    Tr, N, C, P, T = trialR.shape
    # trial-average data
    R = np.mean(trialR, 0)
    # center data
    centers = np.mean(R.reshape((N, -1)), 1)[:, None, None, None]
    R -= centers

    return trialR, R, centers

def tran_dpca(signal, probe, channels, transitions, smooth_window, significance, dPCA_parms={}, raster_fs=None):
    '''
    signal wrapper for dPCA usigg CPN tripplets.
    :param signal: CPN triplets signal
    :param probe: int, over which probe to perform the cPCA
    :param channels: str or [str,], what channels to use
    :param transitions: str or [str.], what context probe transitions to consider
    :param smooth_window: float, smoothing window in ms
    :return: Z, dict of arrays of mean projection into different marginalizations;
             trialZ, dict of arrays of single trial projection into different marginalizations;
             significant_mask
             exp_var, dict of arrays with explained variance
    '''

    triplet_defaults = {'labels': 'ct',
                        'regularizer': 'auto',
                        'n_components': 10,
                        'join': {'ct': ['c', 'ct']}}

    dPCA_parms.update(triplet_defaults)

    full_array, invalid_cp, valid_cp, all_contexts, all_probes = \
        tp.make_full_array(signal, channels=channels, smooth_window=smooth_window, raster_fs=raster_fs)


    raster = tp.extract_sub_arr(probes=probe, context_types=transitions, full_array=full_array,
                                context_names=all_contexts, probe_names=all_probes, squeeze=False )
    raster = raster [..., 100:]  # get only the response to the probe and not the context

    # reorders dimentions from Context x Trial x Neuron x Time  to  Trial x Neuron x Context x Time
    trialR, R, _ = format_raster(raster)

    trialR, R = np.squeeze(trialR), np.squeeze(R)
    Tr, N, C, T = trialR.shape


    dPCA_parms['n_components'] = N if N < dPCA_parms['n_components'] else dPCA_parms['n_components']

    # initializes model
    dpca = dPCA.dPCA(**dPCA_parms)
    dpca.protect = ['t']

    # Now fit the data (R) using the model we just instantiated. Note that we only need trial-to-trial data when we want to
    # optimize over the regularization parameter.
    Z = dpca.fit_transform(R, trialR)

    expt_var = dpca.explained_variance_ratio_.copy()

    # check for significance
    if significance:
        significance_masks = dpca.significance_analysis(R, trialR, axis='t', n_shuffles=100, n_splits=100,
                                                    n_consecutive=1)
    else:
        significance_masks = {}

    # transform in a trial by trial basis
    trialZ = dict()
    for marg in dpca.marginalizations.keys():
        zz = np.empty([Tr, dPCA_parms['n_components'], C, T])
        for rep in range(trialR.shape[0]):
            zz[rep, ...] = dpca.transform(trialR[rep,...], marginalization=marg)
        trialZ[marg] = zz

    dpca.explained_variance_ratio_ = expt_var

    return Z, trialZ, significance_masks, dpca

def signal_transform_triplets_(signal, probe, channels, smooth_window=None, dpca=None):

    signal = signal.rasterize()
    # extract and organizese relevant data from signal
    full_array, invalid_cp, valid_cp, context_names, probe_names = \
        tp.make_full_array(signal, channels=channels, smooth_window=smooth_window)

    raster = tp.extract_sub_arr(probe=probe, context_types='all', full_array=full_array,
                       context_names=context_names, probe_names=probe_names, squeeze=False)

    # get only the probe response
    eps = signal.epochs
    times = eps.loc[eps.name =='C0_P2', ['start', 'end']].values[0,:]
    prb_start_smp =  np.round((times[1] - times[0]) * signal.fs / 2).astype('int16')
    raster = raster[:,:,:,:,prb_start_smp:]

    # get the arrays necesary for dPCA
    trialR, R, centers = format_raster(raster)
    trialR, R = trialR.squeeze(), R.squeeze()

    Tr, N, C, T = trialR.shape
    n_components = N if N < 10 else 10

    # initializes dPCA with
    if dpca is None:
        dpca = dPCA.dPCA(labels='ct', regularizer='auto', n_components=n_components, join={'ct': ['c', 'ct']})
        dpca.protect = ['t']
    else:
        pass

    # Now fit the data (R) using the model we just instantiated. pases trialR for regularization (?)
    dpca.fit(R, trialR)

    # prepares signal raw data by centering equally than training data
    ch_idx = _channel_handler(signal, channels)
    X = signal._data[ch_idx, :] - centers.squeeze()[:,None]

    # dpca canned transformation
    Xt = dpca.transform(X)

    # creates a signal with the transformed data, includes the variance explained by this marginalization
    signals = dict()
    for key, xt in Xt.items():
        new_meta = signal.meta.copy()
        new_meta['expl_var'] = dpca.explained_variance_ratio_[key]
        new_meta['cellid'] = channels
        new_chan_names = [f'{key}{cc+1}' for cc in range(n_components)]
        signals[key] = signal._modified_copy(data=xt, chans=new_chan_names, meta=new_meta)

    return signals

#### plot functions #####

def variance_explained(dpca, ax=None, names=None, colors=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    expl_var = dpca.explained_variance_ratio_

    n_comp = len(next(iter(expl_var.values())))

    bar_bottom = np.zeros(n_comp)
    summed = np.empty(len(expl_var))
    for mm, (marg, arr) in enumerate(expl_var.items()):
        summed[mm] = np.sum(arr) * 100
        # plots the explained variance
        x = np.arange(n_comp) + 1  # the x locations for the groups
        y = np.asarray(arr)*100 # makes into %
        width = 0.35  # the width of the bars: can also be len(x) sequence

        color = colors[mm] if colors is not None else None
        label = names[mm] if names is not None else marg

        ax.bar(x, y, width, bottom=bar_bottom, label=label, color=color)
        ax.set_xticks(range(1,n_comp+1, 4))
        ax.set_xticklabels(range(1,n_comp+1, 4))
        ax.set_ylabel('explained variance (%)')
        bar_bottom += y

    # adds a noise value as whatever var not explained by marginalizations
    noise_frac = np.asarray(100 - np.sum(summed))[None]
    summed = np.concatenate([summed, noise_frac])
    inset = inset_axes(ax, width="50%", height="50%", loc=1)
    explode = np.zeros(len(summed))
    explode[-2] = 0.1 # explodes the second to last fractions, asumes it is the marignaliztion of interest.
    if names is not None: names.append('noise')
    if colors is not None: colors.append('lightgray')
    inset.pie(summed, explode, names, colors, autopct='%1.1f%%')

    return fig, ax, inset

def weight_pdf(dpca, marginalization=None, axes=None, cellnames=None, only_first=False, color=None):

    if marginalization is None:
        marginalization = list(dpca.P.keys())
    elif isinstance(marginalization, str):
        marginalization = [marginalization]


    if axes is None:
        fig, axes = plt.subplots(1, len(marginalization), squeeze=False)
        fig.suptitle('PDF marginalization weights')

    else:
        fig = axes.figure

    axes = np.ravel(axes)

    for mm, marg in enumerate(marginalization):

        if only_first:
            dd = dpca.P[marg][:,0]  # Neurons x Components
            pdf = gkde(dd)
            x = np.linspace(-1, 1, 100, endpoint=False)
            axes[0].plot(x, pdf(x), color=color[mm], linewidth=2)
            axes[0].set_title(marg)
            axes[0].set_xlabel('encoder weight')
            axes[0].set_ylabel('probability density')
            axes[0].scatter(dd, np.zeros(len(dd)), color=color[mm], alpha=0.5)

        else:
            P = dpca.P[marg] # Neurons x Components
            for ii in range(P.shape[-1]):
                dd = P[:, ii]
                pdf = gkde(dd)
                color = 'green' if ii == 0 else 'gray'
                x = np.linspace(-1, 1, 100, endpoint=False)

                axes[mm].plot(x, pdf(x), color=color, linewidth = 2/(ii+1))
                axes[mm].set_title(marg)
                axes[mm].set_xlabel('encoder weight')
                axes[mm].set_ylabel('probability density')

                if ii == 0: axes[mm].scatter(dd, np.zeros(len(dd)), color=color, alpha=1/(ii+1))

                if ii == 0 and cellnames is not None:
                    cellnames = [cell[-4:] for cell in cellnames]
                    ticks = dd.tolist()
                    ticks.extend([-1,1])
                    tick_lables = cellnames.copy()
                    tick_lables.extend([-1,1])
                    # todo diferenciate betwee minor and mayor ticks
                    axes[mm].set_xticks(ticks)
                    axes[mm].set_xticklabels(tick_lables, rotation='vertical')

    return fig, axes



