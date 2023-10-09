import numpy as np

from src.data.rasters import load_site_formated_raster

# ToDo format, and document all functions

def get_diagonalizations(fnArr, distance='L2'):
    psth = fnArr.mean(axis=0, keepdims=True)
    prb_psth = psth.mean(axis=2, keepdims=True)  # average out contexts -> probe centroids
    no_prb_psth = psth - prb_psth
    rep, chn, ctx, prb, tme = no_prb_psth.shape

    # despite having averaged out the probes, since there is an interaction between probe and context
    # we need to keep this into account (?), therefore we have to do a probewise "diagonalization"

    no_prb_diag = np.empty(no_prb_psth.shape)  # parital diagonalization for no-probe data
    for pp, tt in np.ndindex(prb, tme):
        arr = no_prb_psth[..., pp, tt]

        # since disntancese are signless, there are two points in the diagonal for each distance, we want to move
        # our origianl points to closest in the diagonal. find which points are mostly possitive or negative
        # neurons and define the "signed" diagonal target
        sign_flip = np.sign(arr.sum(axis=1, keepdims=True))
        sign_flip[
            sign_flip == 0] = 1  # in the very improbable case of balanced activity, send to the possitive diagonal

        # calculate the distance to preserve
        # all neurons contribute equally (division),
        # how much do we need to give or take from each neuron to reach this equal contribution?? (subtraction)
        if distance == 'L1':
            l1 = np.linalg.norm(arr, ord=1, axis=1, keepdims=True)  # shape 1rep x 1chn x Nctx x
            no_prb_diag[:, :, :, pp, tt] = (l1 * sign_flip / chn) - arr
        elif distance == 'L2':
            l2 = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)  # shape 1rep x 1chn x Nctx x
            no_prb_diag[:, :, :, pp, tt] = np.sqrt(l2 ** 2 / chn) * sign_flip - arr
        else:
            raise ValueError(f"unknow disntance {distance}, use 'L1' or 'L2'")
    return no_prb_diag


def centroid_var(fnArr):
    return (fnArr.mean(axis=0, keepdims=True) ** 2).mean(axis=2, keepdims=True) - \
        fnArr.mean(axis=(0, 2), keepdims=True) ** 2


def cluster_var(fnArr):
    return fnArr.var(axis=0, keepdims=True).mean(axis=2, keepdims=True)


def fano(fnArr, axis=None, keepdims=False):
    return np.var(fnArr, axis=axis, keepdims=keepdims) / np.mean(fnArr, axis=axis, keepdims=keepdims)


def diag_and_scale(fnArr, mode='fano_var', verbose=False, return_problem=False):
    diagArr = fnArr + get_diagonalizations(fnArr)

    #### DANGER !!! full neuron fano makes sense but might generate big artifacts (?)
    TarFano = fano(fnArr, axis=(0, 2, 3, 4), keepdims=True)

    if mode == 'fano_var':
        '''
        this mode is quite convoluted, it manipulates the variance coming from both the context cluster means
        and from the within cluster variance. The advantage of this is that the former has an effect on
        the fano factor on numerator and denominator, while the later only on the numerator.
        This enables some scaling with matches both the original fano factor and variance.
        The only downside is that is numerically unstable due to div0 and sqrt(-), and it slightly shifts
        the diagonalized contexts off the x=y diagonal.
        '''

        TarVar = np.tile(fnArr.var(axis=(0, 2), keepdims=True  # trials and contexts are the sources of var
                                   ).mean(axis=(1), keepdims=True),  # avg acros neurons
                         (1, fnArr.shape[1], 1, 1, 1))  # repeat the avg for all neurons

        ClustVar = cluster_var(diagArr)
        ClustVar[np.isclose(ClustVar, 0)] = 0  # avoids the issue of exploding numbers with tiny donominatosr

        MeanVar = centroid_var(diagArr)
        Mean = diagArr.mean(axis=(0, 2, 3, 4), keepdims=True)

        # here is the magic, Sm and Sc were derived with a bunch of algebra
        # Scaler for the (cluster) means
        if np.any(bads := np.isclose((TarFano * Mean), 0)):
            print(f'div by 0 in TarFano*Means {bads.sum()}')
        Sm = TarVar / (TarFano * Mean)

        # Scaler for the clusters, alas is one for all
        if np.any(bads := (TarVar - Sm ** 2 * MeanVar) < 0):
            print(f'square of negative {bads.sum()}')
        # if np.any(bads:= np.isclose(ClustVar, 0)):
        if np.any(bads := ClustVar == 0):
            print(f'div by 0 in ClustVar {bads.sum()}')
        Sc = np.sqrt((TarVar - (Sm ** 2 * MeanVar)) / ClustVar)

        # troubles with div0 and sqr(-n)
        problem = []
        for S in [Sm, Sc]:
            # problem.append(~np.isfinite(S))
            problem.append(S.copy())
            S[~np.isfinite(S)] = 1

    elif mode == 'mean_var':
        # wiven that diagonalization only move means, the  only change in variance comes from there,
        # to match the orginal centroid variance is our target
        TarVar = np.tile(centroid_var(fnArr).mean(axis=(1), keepdims=True),  # avg acros neurons
                         (1, fnArr.shape[1], 1, 1, 1))  # repeat the avg for all neurons
        MeanVar = centroid_var(diagArr)

        Sm = np.sqrt(TarVar / MeanVar)
        Sc = 1  # leave the clusters alone

        problem = None  # chill, there no problem man

    else:
        raise ValueError(f"unrecognized mode value {mode}, use 'fano_var' or 'mean_var'")

    ctx_psth = diagArr.mean(axis=(0, 3), keepdims=True)  # averages out probe and trials -> context psth
    no_ctx = diagArr - ctx_psth
    diag_scaled = no_ctx * Sc + ctx_psth * Sm

    scaled_fano = fano(diag_scaled, axis=(0, 2, 3, 4)).squeeze()
    scaled_var = diag_scaled.var(axis=(0, 2)).squeeze()

    if verbose:
        print(f'target var={fnArr.var(axis=(0, 2)).squeeze()},'
              f' sum={fnArr.var(axis=(0, 2)).sum()}, fano={TarFano.squeeze()}')
        print(f'scaled var={scaled_var}, sum={scaled_var.sum()}, fano={scaled_fano}')

    if return_problem:
        return diag_scaled, problem
    else:
        return diag_scaled


def load_site_dense_raster(site, part='probe', recache_rec=False, **kwargs):
    """
    wrapper of wrappers, loads raster and diagonalizes it using only the mean variance constrain
    """

    # Load full raster but uses only the probe to fit the PCA.
    # check for existing cache for good measure
    if load_site_formated_raster.check_call_in_cache(site, part=part, recache_rec=recache_rec, **kwargs):
        raster, goodcells = load_site_formated_raster(site, part=part, recache_rec=recache_rec, **kwargs)
    raster = diag_and_scale(raster, mode='mean_var')
    return raster, goodcells
