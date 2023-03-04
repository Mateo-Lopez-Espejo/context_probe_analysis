import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.visualization.palette import TENCOLOR


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


def diag_and_scale(fnArr, mode='fano_var', verbose=False):
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
        for S in [Sm, Sc]:
            S[~np.isfinite(S)] = 1

    elif mode == 'mean_var':
        # wiven that diagonalization only move means, the  only change in variance comes from there,
        # to match the orginal centroid variance is our target
        TarVar = np.tile(centroid_var(fnArr).mean(axis=(1), keepdims=True),  # avg acros neurons
                         (1, fnArr.shape[1], 1, 1, 1))  # repeat the avg for all neurons
        MeanVar = centroid_var(diagArr)

        Sm = np.sqrt(TarVar / MeanVar)
        Sc = 1  # leave the clusters alone

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

    return diag_scaled


######### plotting related to the diagonalizaiton #####


def plot_ctx_clusters(fnArr):
    rep, chn, ctx, prb, tme = fnArr.shape
    assert chn == 2  # can only plot two neurons in the plane
    assert prb == 1  # can only plot one probe
    assert tme == 1  # can only plot one time point

    fig = go.Figure()
    colors = [TENCOLOR[ii % 10] for ii in range(ctx)]

    arrs = [fnArr, fnArr.mean(axis=0, keepdims=True)]
    markersizes = [5, 10]
    opacities = [0.8, 1]

    for arr, ms, op in zip(arrs, markersizes, opacities):
        for cc in range(ctx):
            x = arr[:, 0, cc, 0, 0]
            y = arr[:, 1, cc, 0, 0]
            nreps = x.shape[0]
            if nreps > 1:
                # add some jitter to single trials
                jitter = np.random.uniform(-0.3, 0.3, (nreps, 2))
                x = x + jitter[:, 0]
                y = y + jitter[:, 1]

            _ = fig.add_trace(
                go.Scatter(x=x, y=y,
                           mode='markers', marker=dict(color=colors[cc], size=ms,
                                                       opacity=op),
                           showlegend=False),
            )

    # diagonals
    dd = np.asarray([np.min(fnArr[0, 0, :, 0, 0]), np.max(fnArr[0, 0, :, 0, 0])])
    _ = fig.add_trace(
        go.Scatter(x=dd, y=dd, mode='lines', line=dict(color='black', dash='dot'),
                   name='sign threshold')
    )

    # average acroos all contexts
    grandmeam = fnArr.mean(axis=(0, 2)).squeeze()
    _ = fig.add_trace(
        go.Scatter(x=[grandmeam[0]], y=[grandmeam[1]], mode='markers',
                   marker=dict(color='black', symbol="star-square", size=12),
                   showlegend=False)
    )

    fig.add_vline(0, line_color='black', line_dash='dash')
    fig.add_hline(0, line_color='black', line_dash='dash')

    # axis labels
    _ = fig.update_yaxes(scaleanchor='x', scaleratio=1, title=dict(text='neuron 2 activity (AU)', standoff=0))
    _ = fig.update_xaxes(title=dict(text='neuron 1 activity (AU)', standoff=0))

    return fig


def plot_eg_diag(fnArraList: list):
    fig = make_subplots(1, len(fnArraList), shared_xaxes='all', shared_yaxes='all')

    for cc, fnArr in enumerate(fnArraList):
        traces = plot_ctx_clusters(fnArr)['data']
        _ = fig.add_traces(traces, rows=[1] * len(traces), cols=[cc + 1] * len(traces))

    # zero lines
    fig.add_vline(0, line_color='black', line_dash='dash')
    fig.add_hline(0, line_color='black', line_dash='dash')

    # axis labels
    _ = fig.update_yaxes(scaleanchor='x', scaleratio=1, title=dict(text='neuron 2 activity (AU)', standoff=0),
                         row=1, col=1)
    _ = fig.update_xaxes(title=dict(text='neuron 1 activity (AU)', standoff=0))

    return fig
