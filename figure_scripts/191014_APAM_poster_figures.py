import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import numpy as np
import scipy.stats as sst
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

import itertools as itt
from cycler import cycler

from progressbar import ProgressBar

from cpn_load import load
from reliability import signal_reliability
import cpn_dprime as cpd
import cpn_dPCA as cdPCA
import fancy_plots as cplt
import cpn_LDA as cLDA
from tools import shuffle_along_axis as shuffle
import scipy.stats as sst
import scipy.cluster.hierarchy as sch

from keras.layers import Input, Dense
from keras.models import Model, Sequential

from dPCA import dPCA


def mean_confidence_interval(array, confidence=0.95, axis=0):
    '''
    calculates the mean and confidence interval of an array
    :param array:
    :param confidence:
    :return:
    '''
    n = array.shape[axis]
    m, se, std = np.mean(array, axis=axis), sst.sem(array, axis=axis), np.std(array, axis=axis)
    h = se * sst.t.ppf((1 + confidence) / 2., n - 1) # ToDo check if this formula is adecuate
    return m, h


def cint(array, confidence, ax=None, fillkwargs={}):
    if ax is None:
        fig, ax = plt.subplots()

    mean, h = mean_confidence_interval(array, confidence, axis=1)

    x = np.arange(0, array.shape[0], 1)
    ax.fill_between(x, mean - h, mean + h, **fillkwargs)

    return ax


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def blobplot(arrays, colors=None, ax=None):
    '''
    trial correlation for different stimuli
    :param arrays: list of arrays, each with shape Repetitions x Dimension (e.g. LDA projections)
    :param ax:
    :return:
    '''

    if not isinstance(arrays, list):
        arrays = [arrays]

    if ax is None:
        fig, ax = plt.subplots()

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle']()
    else:
        colors = cycler('color', colors)

    for array, color in zip(arrays, colors):
        cc = color['color']
        X, Y = array[:, 0], array[:, 1]
        ax.scatter(X, Y, color=cc)
        confidence_ellipse(X, Y, ax=ax, n_std=3.0, edgecolor=cc)


def corr_over_times(X):
    '''
    calculates the pearsons correlation coefficient matrix over time, yielding a 3d tensor of correlations
    :param X: nd array with shape Repetition x Cell x Time
    :return: nd array with shape Cell x Cell X Time
    '''

    R, C, T = X.shape
    X_corr = np.empty([C, C, T])
    X_corr[...] = np.nan

    for tt in range(T):
        X_corr[:, :, tt] = np.corrcoef(X[:, :, tt].T)

    return X_corr


def cluster_corr(corr, idx=None, cluster_th=4):
    # if no indexes, defines the new indexes for clustering and performs the clustering
    if idx is None:
        d = sch.distance.pdist(corr)
        L = sch.linkage(d, method='complete')
        ind = sch.fcluster(L, 0.5 * d.max(), 'distance')

        idx = np.argsort(ind)
        clustered = corr[idx[:, None], idx[None, :]]

        # todo implement subclustering

        # unique, counts = np.unique(ind, return_counts=True)
        # counts = dict(zip(unique, counts))
        #
        # i = 0
        # j = 0
        # columns = []
        # for cluster_l1 in set(sorted(ind)):
        #     j += counts[cluster_l1]
        #     sub = corr[i:j, i:j]
        #     if counts[cluster_l1] > cluster_th:
        #         d = sch.distance.pdist(sub)
        #         L = sch.linkage(d, method='complete')
        #         ind = sch.fcluster(L, 0.5 * d.max(), 'distance')
        #         sidx = np.argsort(ind)
        #         sub = sub[sidx[None,:], sidx[:,None]]
        #     cols = sub.columns.tolist()
        #     columns.extend(cols)
        #     i = j
        # df = df.reindex_axis(columns, axis=1)
        #

    # if indexes are given, performs the clustering acording to them
    else:
        clustered = corr[idx[:, None], idx[None, :]]

    return clustered, idx


def zscore2(array):
    """
    hardcoded zscore for this application
    calculates the zscore along the Rep, Stim and Time axis. doing it independently for each cell
    :param array: nd array of shape Rep x Cell x Stimulus(context) x Time
    :return: zscored array of the same shape as input
    """
    means = np.mean(array, axis=(0, 2, 3))[None, :, None, None]
    stds = np.std(array, axis=(0, 2, 3))[None, :, None, None]

    zscore = np.nan_to_num((array - means) / stds)

    return zscore


def autoencoder(array, ncomp=5):
    model = Sequential()
    model.add(Dense(ncomp, input_shape=array.shape[1:], activation='linear'))
    model.add(Dense(array.shape[1:], activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(array, array, batch_size=array.shape[0], epoch=100)
    return model


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',
                  '#984ea3', '#999999', '#e41a1c', '#dede00']

trans_color_map = {'silence': 'blue',
                   'continuous': 'orange',
                   'similar': 'green',
                   'sharp': 'brown'}

# loads some test data
meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 0, # ms
        'raster_fs': 10,
        'transitions' : ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot' : [2,3,5,6],
        'significance': False}

analysis_name = 'LDA_dprime'


analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])


code_to_name = {'t': 'Probe', 'ct': 'Context'}


site = 'AMT029a'
probe = 5
# for site, probe in zip(['AMT029a', 'ley070a'],[5,2]):

recs = load(site)
rec = recs['trip0']
sig = rec['resp']

# calculates response realiability and select only good cells to improve analysis
r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
goodcells = goodcells.tolist()




# for site, probe in zip(['AMT029a', 'ley070a'],[5,2]):
# iterates over pairs of context transistions to calculate discriminability:
# for c0, c1 in itt.combinations(range(4),2):
# c0 = 1
# c1 = 3

####### OLD dPCA ct marginalization projection as a refference point.

# transitions = meta['transitions']
transitions = ['continuous', 'sharp']
# transitions = meta['transitions']
colors = [trans_color_map[trans] for trans in transitions]

Z, trialZ, significance_masks, dpca = cdPCA.tran_dpca(sig, probe, channels=goodcells, transitions=transitions,
                                             smooth_window=meta['smoothing_window'], significance=meta['significance'],
                                             raster_fs=meta['raster_fs'])

# repeats the same transformation matrix over all the time bins
dpca_axes = np.tile(dpca.D['ct'][:,:,None], (1,1,trialZ['ct'].shape[-1]))

all_dPCA_proj = [trialZ['ct'][:, 0, ctx, :] for ctx in range(trialZ['ct'].shape[2])]


########################################################################################################################
# figure 1: Shows the raw data, i.e.the projection
# gets the real data raster (no dim reduction) to calculate single cell, population independent d'
raster = cdPCA.raster_from_sig(sig, probe, channels=goodcells, transitions=transitions,
                               smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'])

# trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
trialR, R, _ = cdPCA.format_raster(raster)
trialR = trialR.squeeze() # squeezes out probe
R, C, S, T = trialR.shape

# calculatese all LDA projectios
aca = [trialR[:, :, ctx, :] for ctx in range(S)] #aca: all context arrays, with shape Rep x Cell x Time
lda_axes2 = cLDA.get_LDA_ax_over_timev2(aca, N=1)
all_LDA_proj = [cLDA.transform_over_time(a, lda_axes2) for a in aca]


# plots figure

fig, axes = plt.subplots(1,2)
axes = np.ravel(axes)

for ii, (LDA_proj, dPCA_proj, color) in enumerate(zip(all_LDA_proj, all_dPCA_proj, colors)):
    axes[0].plot(dPCA_proj.T, color=color, alpha=0.1)
    axes[0].plot(np.mean(dPCA_proj, axis=0), color=color)
    label = f"{transitions[ii]}"
    axes[1].plot(LDA_proj.squeeze().T, color=color, alpha=0.1)
    axes[1].plot(np.mean(LDA_proj.squeeze(), axis=0), color=color, label=label)
axes[0].set_title('dPCA projection')
axes[1].set_title('LDA projection')
axes[1].legend()


########################################################################################################################
# figure 2: pairwise dprime calculation, with floor and ceiling, for the LDA and dPCA projections
c0, c1 = 1, 3

# LDA approach
# calculates real dprime
LDA_dprime = cpd.dprime(all_LDA_proj[c0].squeeze(), all_LDA_proj[c1].squeeze())

# calculates both the floor and cealing d' for 1000 random shuffles/simulations
nreps = 1000
LDA_floor_d = np.empty((nreps, LDA_dprime.shape[0]))
LDA_ceil_d = np.empty((nreps, LDA_dprime.shape[0]))

ctx_shuffle = trialR.copy()
pbar = ProgressBar()
for ii in pbar(range(nreps)):

    # dprime celing: normal simulation of the data, projection and dprime calculation
    sim_trialR = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0), size=[R, C, S ,T]).squeeze()
    sim_aca = [sim_trialR[:, :, ctx, :] for ctx in range(S)]
    sim_LDA_proj = [cLDA.transform_over_time(a, lda_axes2) for a in sim_aca]
    LDA_ceil_d[ii, :] = cpd.dprime(sim_LDA_proj[c0].squeeze(), sim_LDA_proj[c1].squeeze())

    # dprime floor: shuffle context(dim2) identity by trial(dim0), projection and dprime calculation
    ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)

    shuf_aca = [ctx_shuffle[:, :, ctx, :] for ctx in range(S)]  # aca: all context arrays, with shape Rep x Cell x Time
    shuf_LDA_axes = cLDA.get_LDA_ax_over_timev2(shuf_aca, N=1)
    shuf_LDA_proj = [cLDA.transform_over_time(a, shuf_LDA_axes) for a in shuf_aca]

    LDA_floor_d[ii, :] = cpd.dprime(shuf_LDA_proj[c0].squeeze(), shuf_LDA_proj[c1].squeeze())



# dPCA approach
# calculates real dprime
dPCA_dprime = cpd.dprime(all_dPCA_proj[c0].squeeze(), all_dPCA_proj[c1].squeeze())

# calculates both the floor and cealing d' for 1000 random shuffles/simulations
nreps = 1000
dPCA_floor_d = np.empty((nreps, dPCA_dprime.shape[0]))
dPCA_ceil_d = np.empty((nreps, dPCA_dprime.shape[0]))

ctx_shuffle = trialR.copy()
pbar=ProgressBar()
for ii in pbar(range(nreps)):
    # dprime celing: normal simulation of the data, projection and dprime calculation
    sim_trialR = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0), size=[R, C, S, T]).squeeze()
    sim_aca = [sim_trialR[:, :, ctx, :] for ctx in range(S)]
    sim_dPCA_proj = [cLDA.transform_over_time(a, dpca_axes) for a in sim_aca]
    dPCA_ceil_d[ii, :] = cpd.dprime(sim_dPCA_proj[c0][:,0,:], sim_dPCA_proj[c1][:,0,:])

    # dprime floor: shuffle context(dim2) identity by trial(dim0), projection and dprime calculation
    ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)
    # trial-average data
    shuf_R = np.mean(ctx_shuffle, 0)
    # center data per neuron i.e. first dime
    centers = np.mean(shuf_R.reshape((shuf_R.shape[0], -1)), axis=1)[:, None, None]
    shuf_R -= centers
    # calculates dpca and projections
    _, shuf_trialZ, _, _ = cdPCA.trials_dpca(shuf_R, ctx_shuffle, significance=False)
    shuf_dPCA_proj = [shuf_trialZ['ct'][:, 0, ctx, :] for ctx in range(shuf_trialZ['ct'].shape[2]) ]
    dPCA_floor_d[ii, :] = cpd.dprime(shuf_dPCA_proj[c0].squeeze(), shuf_dPCA_proj[c1].squeeze())


# plots figure

fig, axes = plt.subplots(1,2, sharey=True)
axes = np.ravel(axes)

ci =  0.99
axes[0].plot(np.mean(LDA_floor_d, axis=0), color='red', alpha=1, label='floor')
cint(LDA_floor_d.T, ci, ax=axes[0], fillkwargs={'alpha': 0.5, 'color':'red'})
# ceiling
axes[0].plot(np.mean(LDA_ceil_d, axis=0), color='green', alpha=1, label='ceiling')
cint(LDA_ceil_d.T, ci, ax=axes[0], fillkwargs={'alpha': 0.5, 'color':'green'})
# real dprime
axes[0].plot(LDA_dprime, color='black', linestyle=':', label='d prime')
axes[0].set_title('LDA analysis')



axes[1].plot(np.mean(dPCA_floor_d, axis=0), color='red', alpha=1, label='floor')
cint(dPCA_floor_d.T, ci, ax=axes[1], fillkwargs={'alpha': 0.5, 'color':'red'})
# ceiling
axes[1].plot(np.mean(dPCA_ceil_d, axis=0), color='green', alpha=1, label='ceiling')
cint(dPCA_ceil_d.T, ci, ax=axes[1], fillkwargs={'alpha': 0.5, 'color':'green'})
# real dprime
axes[1].plot(dPCA_dprime, color='black', linestyle='--', label='d prime')
axes[1].set_title('dPCA analysis')
axes[1].legend()








