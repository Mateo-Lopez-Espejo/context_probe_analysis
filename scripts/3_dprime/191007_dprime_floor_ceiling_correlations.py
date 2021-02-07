import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import numpy as np
from sklearn.decomposition import PCA

from cycler import cycler

import src.data.rasters
from src.data.load import load
from src.metrics.reliability import signal_reliability
from src.metrics import dprime as cpd
from src.data import LDA as cLDA, dPCA as cdPCA
from src.utils.tools import shuffle_along_axis as shuffle
import scipy.stats as sst
import scipy.cluster.hierarchy as sch




"""
I have found a ceiling for the d' by using an optimal LDA decoder that adjusts over time (calculated independently for 
each time bin). 

the d' calculated with a dPCA projection is somehowe lower but has a similar shape

Now I need to define a floor value of d' values coming from random chance to properly asses the reduction over time of 
real d' and a lowe asimptote once contextual effects are extinguished. 

It is worth seen how correlations between neurons are changing over time and howe this affects d'
This is particularly important given that it seems that correlations themselves are the main source of discriminability
and not independent gain change, which have been acounded for by virtue of the dPCA and LDA (??? not sure about this)

for this purpose it is worth to make "blob plots" of the trial distributionse between pairs of contexts 

"""


def mean_confidence_interval(array, confidence=0.95, axis=0):
    '''
    calculates the mean and confidence interval of an array
    :param array:
    :param confidence:
    :return:
    '''
    n = array.shape[axis]
    m, se = np.mean(array, axis=axis), sst.sem(array, axis=axis) #ToDo check if this formula is adecuate
    h = se * sst.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def cint(array, confidence, ax=None, fillkwargs={}):

    if ax is None:
        fig, ax = plt.subplots()


    mean, h = mean_confidence_interval(array, confidence, axis=1)

    x = np.arange(0,array.shape[0],1)
    ax.fill_between(x, mean-h, mean+h, **fillkwargs)

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
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
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
        ax.scatter(X,Y, color=cc)
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
        X_corr[:,:, tt] = np.corrcoef(X[:, :, tt].T)

    return X_corr


def cluster_corr(corr, idx=None, cluster_th=4):

    # if no indexes, defines the new indexes for clustering and performs the clustering
    if idx is None:
        d = sch.distance.pdist(corr)
        L = sch.linkage(d, method='complete')
        ind = sch.fcluster(L, 0.5 * d.max(), 'distance')

        idx = np.argsort(ind)
        clustered = corr[idx[:,None], idx[None,:]]

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
        clustered = corr[idx[:,None], idx[None,:]]

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


# def autoencoder(array, ncomp = 5):
#     model = Sequential()
#     model.add(Dense(ncomp, input_shape=array.shape[1:], activation='linear'))
#     model.add(Dense(array.shape[1:], activation='linear'))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     model.fit(array, array, batch_size=array.shape[0], epoch=100)
#     return model


# loads some test data
meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 0, # ms
        'raster_fs': 10,
        'transitions' : ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot' : [2,3,5,6],
        'significance': False}

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

####### OLD dPCA ct marginalization projection as a refference point.

# transitions = meta['transitions']
transitions = ['continuous', 'sharp']
Z, trialZ, significance_masks, dpca = cdPCA.tran_dpca(sig, probe, channels=goodcells, transitions=transitions,
                                             smooth_window=meta['smoothing_window'], significance=meta['significance'],
                                             raster_fs=meta['raster_fs'])

# iterates over pairs of context transistions to calculate discriminability:
# for c0, c1 in itt.combinations(range(4),2):
c0 = 1
c1 = 3

########
# gets the real data raster (no dim reduction) to calculate single cell, population independent d'
raster = src.data.rasters.raster_from_sig(sig, probe, channels=goodcells, contexts=meta['transitions'],
                                          smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'])

# trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
trialR, _, _ = cdPCA.format_raster(raster)
trialR = trialR.squeeze() # squeezes out probe
trialR = trialR[:, :,(c0, c1), :] # selects only the two contexts to compare
R, C, S, T = trialR.shape

# calculates the collection of LDA projections using the real data
x = trialR[:, :, 0, :]
y = trialR[:, :, 1, :]

LDA_proj, lda_axes = cLDA.fit_transform_over_time(trialR, 1)

# calcualtes real d prime over the first LDA projection
dprime = cpd.dprime(LDA_proj[:,:,0,:].squeeze(), LDA_proj[:,:,1,:].squeeze())

# calculates both the floor and cealing d' for 1000 random shuffles/simulations
nreps = 1000

floor_d = np.empty((nreps, dprime.shape[0]))
ceil_d = np.empty((nreps, dprime.shape[0]))

ctx_shuffle = trialR.copy()
for ii in range(nreps):

    # dprime celing: normal simulation of the data, projection and dprime calculation
    trial_sim = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0), size=[R, C, S ,T]).squeeze()
    sim_proj = cLDA._recover_dims(cLDA.transform_over_time(cLDA._reorder_dims(trial_sim), lda_axes))
    ceil_d[ii, :] = cpd.dprime(sim_proj[:,:,0,:].squeeze(), sim_proj[:,:,1,:].squeeze())

    # dprime floor: shuffle context(dim2) identity by trial(dim0), projection and dprime calculation
    ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)
    shuf_proj, shuf_lda_axes = cLDA.fit_transform_over_time(ctx_shuffle, 1)
    floor_d[ii, :] = cpd.dprime(shuf_proj[:,:,0,:].squeeze(), shuf_proj[:,:,1,:].squeeze())

# # plots the floor and ceiling confidence intervals
# fig, ax = plt.subplots()
# # floor
# ax.plot(np.mean(floor_d, axis=0), color='red', alpha=1, label='floor')
# _cint(floor_d.T, 0.95, ax=ax, fillkwargs={'alpha': 0.5, 'color':'red'})
# # ceiling
# ax.plot(np.mean(ceil_d, axis=0), color='green', alpha=1, label='ceiling')
# _cint(ceil_d.T, 0.95, ax=ax, fillkwargs={'alpha': 0.5, 'color':'green'})
# # real dprime
# ax.plot(dprime, color='black', label='d prime')
# ax.legend()


# blob plot! 1. get the top two axis for LDA  2. project the single trials into 2d space   3. plot, watch learn
LDA_proj2, lda_axes2 = cLDA.fit_transform_over_time(trialR, 2)

# what if I substract the PSTH from the single trials before correlations? this should eliminate correlations produced
# bi common response to the stimuli

remove_PSTH = True

if remove_PSTH:
    X_noPSTH = x - np.mean(x, axis=0)[None, :, :]
    Y_noPSTH = y - np.mean(x, axis=0)[None, :, :]
else:
    X_noPSTH = x
    Y_noPSTH = y


# calculates correlations between different contexts
# zscore = zscore2(trialR)

X_corr = np.nan_to_num(corr_over_times(X_noPSTH), copy=False, nan=0)
Y_corr = np.nan_to_num(corr_over_times(Y_noPSTH), copy=False, nan=0)

# get a linearized version of the upper triangle sans the diagonal fo the corrleations matrixes, then performs
# PCA over time. i.e. what sets of cell correlations are changing together the most
triu_idx = np.triu_indices(X_corr.shape[0], k=1)
flat_X_corr = X_corr[triu_idx]
flat_Y_corr = Y_corr[triu_idx]


Xpca = PCA(n_components=10)
X_corr_PCA = Xpca.fit_transform(flat_X_corr.T) # Samples (Time) x Features (Cells pairs)

Ypca = PCA(n_components=10)
Y_corr_PCA = Ypca.fit_transform(flat_Y_corr.T) # Samples (Time) x Features (Cells pairs)
Y_corr_PCA = Xpca.transform(flat_Y_corr.T) #transforms with the X corr map

corr_diff = X_corr_PCA - Y_corr_PCA

# # plot variace explained and components, the components are really difficult to read.
# fig, axes = plt.subplots(2,2)
# axes = np.ravel(axes)
# axes[0].plot(np.cumsum(Xpca.explained_variance_ratio_), marker='o')
# axes[0].set_title('var explained X_corr_PCA')
# axes[1].imshow(Xpca.components_, aspect='auto', cmap='inferno')
# axes[1].set_title('pca components X_cor_PCA')
# axes[1].set_ylabel('component')
# axes[1].set_xlabel('correlation pairs')
#
# axes[2].plot(np.cumsum(Ypca.explained_variance_ratio_), marker='o')
# axes[2].set_title('var explained Y_corr_PCA')
# axes[3].imshow(Ypca.components_, aspect='auto', cmap='inferno')
# axes[3].set_title('pca components Y_corr_PCA')
# axes[3].set_ylabel('component')
# axes[3].set_xlabel('correlation pairs')

fig, axes = plt.subplots(3, 10)

for tt in range(X_corr.shape[2]):
    axes[0,tt].imshow(X_corr[:,:,tt], aspect='auto', cmap='inferno')
    axes[1,tt].imshow(Y_corr[:,:,tt], aspect='auto', cmap='inferno')

    axes[2,tt].imshow(corr_diff[tt,:][:,None], aspect='auto', cmap='inferno')
fig.suptitle(f'remove PSTH {remove_PSTH}')


# Todo sorted correlation matrices, all matrices over time are sortede following the same map as the matrix correspondent
# to the time bin with the top dprime
ref_bin = np.where(dprime == np.min(dprime))[0][0]
# ref_bin = 0


_, clust_idx = cluster_corr(X_corr[:,:, ref_bin])

X_corr_clust = X_corr[clust_idx[:,None], clust_idx[None,:]]
Y_corr_clust = Y_corr[clust_idx[:,None], clust_idx[None,:]]

diff_corr_clust = X_corr_clust - Y_corr_clust

fig, axes = plt.subplots(3, X_corr.shape[2])

for tt in range(X_corr.shape[2]):
    axes[0, tt].matshow(X_corr_clust[..., tt])
    axes[1, tt].matshow(Y_corr_clust[..., tt])
    axes[2, tt].matshow(diff_corr_clust[..., tt])
    pass


plt.figure()
plt.plot(dprime)



# plots everything together
fig, axes = plt.subplots(2,3)
axes = np.ravel(axes)

time_bin = 6

# firs ax: 1dim lda projection of single trials plus real dPrime
axes[0].plot(LDA_proj[:,:,0,:].squeeze().T, color='blue', alpha=0.3)
axes[0].plot(LDA_proj[:,:,1,:].squeeze().T, color='orange', alpha=0.3)
axes[0].plot(dprime, color='black')
axes[0].axvline(time_bin, color='black', linestyle='--')

# Second ax: real d' with the floor and ceiling confidence intervals
ci =  0.95
axes[1].plot(np.mean(floor_d, axis=0), color='red', alpha=1, label='floor')
cint(floor_d.T, ci, ax=axes[1], fillkwargs={'alpha': 0.5, 'color':'red'})
# ceiling
axes[1].plot(np.mean(ceil_d, axis=0), color='green', alpha=1, label='ceiling')
cint(ceil_d.T, ci, ax=axes[1], fillkwargs={'alpha': 0.5, 'color':'green'})
# real dprime
axes[1].plot(dprime, color='black', label='d prime')
axes[1].axvline(time_bin, color='black', linestyle='--')
axes[1].legend()

# third ax: blob plot of a time slice
blobplot([LDA_proj2[:,:,0,time_bin], LDA_proj2[:,:,1,time_bin]],
         colors=['blue', 'orange'], ax=axes[2])

axes[3].imshow(X_corr[..., time_bin], aspect='auto', cmap='coolwarm')
axes[3].set_title('blue context correlations')
axes[4].imshow(Y_corr[..., time_bin], aspect='auto', cmap='coolwarm')
axes[4].set_title('oragen context correlations')


# todo cell correlation profile (trial mean) between time bins (this is stehphen confusing approach)




