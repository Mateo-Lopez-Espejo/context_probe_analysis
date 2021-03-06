import numpy as np
from functools import partial
import collections as col

from dPCA import dPCA

from src.data.rasters import raster_from_sig
from src.utils.cpp_parameter_handlers import _channel_handler

from src.data import rasters as rt


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


def _cpp_dPCA(R, trialR, dPCA_parms={}):
    '''
    dPCA over data arrays
    :param R: ndarray. categories (mean) with shape Neuron x Context x TimeBin or Neuron x Context x Probe x TimeBin
    :param trialR: ndarray. raw data with shape Trial x Neuron x Context x TimeBin or
    Trial x Neuron x Context x Probe x TimeBin
    :param dPCA_parms: furthe dPCA parameters to be passed to the function call
    :return: Z, dict of arrays of mean projection into different marginalizations;
             trialZ, dict of arrays of single trial projection into different marginalizations;
             significant_mask
             exp_var, dict of arrays with explained variance
    '''

    single_probe_defaults = {'labels': 'ct',
                        'regularizer': 'auto',
                        'n_components': 10,
                        'join': {'ct': ['c', 'ct']}}

    all_probes_defaults = {'labels': 'cpt',
                             'regularizer': 'auto',
                             'n_components': 10,
                             'join': {'ct' : ['c','ct'], 'pt':['p', 'pt'], 'cpt':['cp', 'cpt']}}

    if len(trialR.shape) == 4:
        dPCA_parms.update(single_probe_defaults)
        Tr, N, C, T = trialR.shape
    elif len(trialR.shape) == 5:
        dPCA_parms.update(all_probes_defaults)
        Tr, N, C, P, T = trialR.shape
    else:
        raise ValueError('trialR has an uncompatible number of dimensions')

    dPCA_parms['n_components'] = N if N < dPCA_parms['n_components'] else dPCA_parms['n_components']

    # initializes model
    dpca = dPCA.dPCA(**dPCA_parms)
    dpca.protect = ['t']

    # Now fit the data (R) using the model we just instantiated. Note that we only need trial-to-trial data when we want to
    # optimize over the regularization parameter.
    Z = dpca.fit_transform(R, trialR)
    expt_var = dpca.explained_variance_ratio_.copy()

    # transform in a trial by trial basis
    trialZ = dict()
    new_trial_shape = list(trialR.shape)
    new_trial_shape[1] = dPCA_parms['n_components']
    for marg in dpca.marginalizations.keys():
        zz = np.empty(new_trial_shape)
        for rep in range(trialR.shape[0]):
            zz[rep, ...] = dpca.transform(trialR[rep,...], marginalization=marg)
        trialZ[marg] = zz

    dpca.explained_variance_ratio_ = expt_var

    return Z, trialZ, dpca

def variance_captured(dPCA, R):
    """
    calculates the variance captured by comparing the data reconstruction i.e. decode(encode(X))) using different sets
    of components. Based on the approach in the matlab dPCA implementation. See 'dpca_explainedVariance.m'
    :param dPCA: fitted dpca object
    :param R: nd array with dimensions Unit x Context x (Probe) x Time
    :return:
    """

    R = dPCA._zero_mean(R)
    total_variance = np.sum((R - np.mean(R)) ** 2)

    marginals = dPCA._marginalize(R)
    total_marginalized_var = {marg: np.sum(arr**2) for marg, arr in marginals.items()}

    # concatenates arrays across marginalizations
    D = list()
    P = list()
    comp_id = list()
    for marg in dPCA.marginalizations.keys():
        D.append(dPCA.D[marg])
        P.append(dPCA.P[marg])
        comp_id.extend([f'{marg}_{comp}' for comp in range(dPCA.n_components)])
    D = np.concatenate(D, 1)
    P = np.concatenate(P, 1)
    R = R.reshape([R.shape[0], -1]) # concatenates and collapses all condition dimensions

    # calculates variance captured for each marginalization for each component
    dpc_var = np.empty(len(marginals) * dPCA.n_components)
    for comp in range(len(marginals) * dPCA.n_components):
        z = R - P[:,[comp]] @ D[:,[comp]].T @ R
        dpc_var[comp] = 100 - np.sum(z**2) / total_variance * 100

    order = np.argsort(dpc_var)
    D = D[:, order[::-1]]
    P = P[:, order[::-1]]
    dpc_var = dpc_var[order[::-1]]
    comp_id = [comp_id[o] for o in order[::-1]]

    Z = D.T @ R
    cum_var = np.empty([len(comp_id)])
    marg_var = col.defaultdict(partial(np.empty, len(comp_id)))
    for comp in range(len(marginals) * dPCA.n_components):
        cum_var[comp] = 100 - np.sum((R - P[:,:comp+1] @ Z[:comp+1,:])**2) / total_variance * 100
        for marg, Xmarg in marginals.items():
            ZZ = Xmarg - P[:,[comp]] @ D[:,[comp]].T @ Xmarg
            marg_var[marg][comp] = (total_marginalized_var[marg] - np.sum(ZZ**2)) /total_variance * 100

    total_marginalized_var = {marg: var/total_variance * 100 for marg, var in total_marginalized_var.items()}

    return cum_var, dpc_var, marg_var, total_marginalized_var, comp_id

def tran_dpca(signal, probe, channels, transitions, smooth_window, dPCA_parms={}, raster_fs=None,
              part='probe', zscore=False):
    '''
    signal wrapper for dPCA using CPN triplets.
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

    # gets a raster, specific for a certain probe and collection of transitions
    raster = raster_from_sig(signal, probe, channels, transitions, smooth_window, raster_fs, part, zscore)

    # reorders dimentions from Context x Trial x Neuron x Time  to  Trial x Neuron x Context x Time
    trialR, R, _ = format_raster(raster)
    trialR, R = np.squeeze(trialR), np.squeeze(R)

    # calculates dPCA
    Z, trialZ, dpca = _cpp_dPCA(R, trialR, dPCA_parms=dPCA_parms)

    return Z, trialZ, dpca


def transform_trials(dpca, trial_array):

    marginalizations = list(dpca.marginalizations.keys())
    ncomp = dpca.D[marginalizations[0]].shape[-1]
    Tr, N, C, T = trial_array.shape

    trialZ = dict()
    for marg in dpca.marginalizations.keys():
        zz = np.empty([Tr, ncomp, C, T])
        for rep in range(trial_array.shape[0]):
            zz[rep, ...] = dpca.transform(trial_array[rep,...], marginalization=marg)
        trialZ[marg] = zz

    return trialZ

# simple wrappers
# wrappers to be treated equally as those funcitons in cpn_LDA. asumes a lot of

def transform(trialR, transformation):
    """
    transforms each trial of trialR into its 1dim projection.
    :param trialR: nd-array with shape Rep x Unit x Context x Time
    :param transformation: nd-array with shape Unit x PC x Time
    :return: nd-array with shape Rep x PC x Context x Time
    """

    # reorders axes to keep units as first dimension, collapses all other dimensions together, then performs the
    # dot product, finally reshapes and transposes back into the desired dimension.

    R, U, C, T = trialR.shape
    Pc = transformation.shape[1]
    neworder = [1,0,2,3]

    projection = np.dot(transformation[:,:,0].T, trialR.transpose(neworder).reshape((U,-1))
                        ).reshape((Pc, R, C, T)).transpose(np.argsort(neworder)).squeeze()

    return projection

def fit_transform(R, trialR, dPCA_params={}):
    '''
    wrapper of dPCA. Uses R to fit the transformation and then projects trialR into the new space.
    :param R: ndarray, shape Cells x Contexts x Time
    :param trialR: ndarray, shape Repetitions x Cells x Contexts x Time
    :param dPCA_params:
    :return: first dPC ndarray (Resp x Ctx x Time), transformation function ndarray (Cells x dPCs x Time)
    '''
    Re, C, S, T = trialR.shape
    _, dPCA_projection, dpca = _cpp_dPCA(R, trialR, dPCA_parms=dPCA_params)
    dPCA_projection = dPCA_projection['ct'][:, 0, ...]
    dPCA_transformation = np.tile(dpca.D['ct'][:, 0][:, None, None], [1, 1, T])

    return dPCA_projection, dPCA_transformation
