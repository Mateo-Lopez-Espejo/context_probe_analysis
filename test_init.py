import cpp_dispersion as cdisp
import math
import numpy as np
import itertools as itt
from sklearn.metrics.pairwise import pairwise_distances

def _window_ndim_MSD(working_window):
    '''
        calculates mean of the pairwise Mean Standard Difference of the PSTHs (i.e. collapsed repetitions) between contexts
        for an array of shape Repetition x Cell x Context x WindowTimeBin
        :param working_window: 4D array has dimensions Repetition x Cell x Context x WindowTimeBin
        :return: float pvalue
        '''

    def _working_window_MSD(working_window):
        # input array should have shape Repetition x Cell x Context x WindowTimeBin
        # calculates PSTH i.e. mean across repetitions
        psth = working_window.mean(axis=0)  # dimentions Cell x Context x WindowTime
        # initializes array to hold the calculated metric for all stim_num pairs combinations
        combs = int(math.factorial(psth.shape[1]) / (2 * math.factorial(psth.shape[1] - 2)))
        msd_values = np.empty(combs)
        # iterates over every pair of contexts
        for ctx, (ctx1, ctx2) in enumerate(itt.combinations(range(psth.shape[1]), 2)):
            # compares the psth difference for each cell
            cell_sqr_dif = (psth[:, ctx1, :] - psth[:, ctx2, :])**2
            # sums across cells, means across time
            msd_values[ctx] = np.nanmean(np.sum(cell_sqr_dif,axis=0),axis=0)

        mean_r = msd_values.mean()
        return mean_r

    # 1. calculates the mean of the pairwise correlation coefficient between all differente contexts
    obs_msd = _working_window_MSD(working_window)

    # 2. shuffles across repetitions a contexts
    # collapses repetition and stim_num together
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

    return pvalue


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

