import itertools as itt
import re

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import make_interp_spline
from scipy.stats import sem
from sklearn.decomposition import PCA

from IPython.display import Image

import src.models.modelnames as mns
from nems.db import batch_comp
from src.data.load import get_batch_ids
from src.data.rasters import load_site_formated_raster
from src.dim_redux.PCA import load_site_formated_PCs
from src.metrics.consolidated_tstat import tstat_cluster_mass
from src.metrics.delta_fr import pairwise_delta_FR
from src.metrics.significance import _significance
from src.models.param_tools import get_population_weights, get_strf, \
    get_population_influence, get_pred_err, model_independence_comparison, \
    load_cell_formated_resp_pred
from src.visualization.palette import (TENCOLOR, FOURCOLOR, hex_to_rgb,
                                       AMPCOLOR, DURCOLOR, Green, Purple, Grey,
                                       Yellow, Teal, Brown, Red)
from src.visualization.utils import squarefy, square_rows_cols


def plot_PSTH(fnArr, time, y0=None, CI=None, CI_opacity=0.2, name=None,
              showlegend=True, fig=None, **line_kwargs):
    """
    fnArr: 2dim array with dims Trials x Time
    """
    line_defaults = dict(color="#000000", width=3)
    line_defaults.update(line_kwargs)

    # expands dimensions to work with vectors, forces no CI
    if fnArr.ndim == 1:
        fnArr = fnArr[np.newaxis, :]
        CI = None

    if fig == None:
        fig = go.Figure()

    psth = np.mean(fnArr, axis=0)

    # If trials available, plots CI first so it lays behind PSTH
    if CI != None and fnArr.shape[0] > 1:
        if CI == 'std':
            err = np.std(fnArr, axis=0)
        elif CI == 'sem':
            err = sem(fnArr, axis=0)
        else:
            raise ValueError(
                f"Unknown CI value {CI}. Use None, 'std' or 'sem'")

        # here without y0
        x, y = squarefy(time, psth)
        _, yerr = squarefy(time, err)

        rgb = hex_to_rgb(line_defaults['color'])  # tuple
        fill_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {CI_opacity})'
        # transparent line in case its width is changed later outside this func
        line_color = 'rgba(0,0,0,0)'

        _ = fig.add_trace(go.Scatter(x=x, y=y + yerr, mode='lines',
                                     line=dict(color=line_color, width=0),
                                     showlegend=False))
        _ = fig.add_trace(go.Scatter(x=x, y=y - yerr, mode='lines',
                                     line=dict(color=line_color, width=0),
                                     fill='tonexty', fillcolor=fill_color,
                                     showlegend=False))

    # plots PSTH second so it lays over CI
    x, y = squarefy(time, psth, y0=y0)
    _ = fig.add_trace(
        go.Scatter(x=x, y=y, mode='lines', line=line_defaults, name=name,
                   showlegend=showlegend))

    return fig


def plot_raster(fnArr, time, y0=0, name=None, showlegend=True, fig=None,
                **marker_kwargs):
    y, x = np.where(fnArr > 0)
    fs = 1 / np.mean(np.diff(time))
    x = (x / fs) + time[0]
    y += y0

    marker_defaults = dict(color='#000000', opacity=0.5, size=5)

    marker_defaults.update(marker_kwargs)

    if fig == None:
        fig = go.Figure()

    _ = fig.add_trace(
        go.Scatter(x=x, y=y, mode='markers', marker=marker_defaults, name=name,
                   showlegend=showlegend))

    return fig


def confidence_ellipse(x, y, n_std=1.96, size=100, **kwargs):
    """
    Get the covariance confidence ellipse of *x* and *y*.
    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    size : int
        Number of points defining the ellipse
    Returns
    -------
    plotly trace of the ellipse

    References (H/T)
    ----------------
    https://gist.github.com/dpfoose/38ca2f5aee2aea175ecc6e599ca6e973
    https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    theta = np.linspace(0, 2 * np.pi, size)
    ellipse_coords = np.column_stack(
        [ell_radius_x * np.cos(theta), ell_radius_y * np.sin(theta)])
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    x_scale = np.sqrt(cov[0, 0]) * n_std
    x_mean = np.mean(x)

    # calculating the stdandard deviation of y ...
    y_scale = np.sqrt(cov[1, 1]) * n_std
    y_mean = np.mean(y)

    translation_matrix = np.tile([x_mean, y_mean],
                                 (ellipse_coords.shape[0], 1))
    rotation_matrix = np.array([[np.cos(np.pi / 4), np.sin(np.pi / 4)],
                                [-np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    scale_matrix = np.array([[x_scale, 0], [0, y_scale]])
    ellipse_coords = ellipse_coords.dot(rotation_matrix).dot(
        scale_matrix) + translation_matrix

    trace = go.Scatter(x=ellipse_coords[:, 0], y=ellipse_coords[:, 1],
                       mode='lines', **kwargs)
    return trace


def plot_raw_pair_array(fnArr, cellids, cellid, contexts, probe, raster_fs,
                        part, prb_use_ctx_color=True,
                        hightlight_difference=False, CI='sem', CI_opacity=0.5,
                        mode='PSTH', dash='solid', showlegend=True,
                        color_palette=FOURCOLOR):
    eg_raster = fnArr[:, cellids.index(cellid), :, probe - 1, :]

    nreps, _, nsamps = eg_raster.shape
    duration = nsamps / raster_fs

    if part == 'all':
        time = np.linspace(0 - duration / 2, duration / 2, nsamps,
                           endpoint=False)
        halfs = [np.s_[:int(nsamps / 2)], np.s_[int(nsamps / 2):]]
    elif part == 'probe':
        # asumes the raste has already been sliced before passing
        time = np.linspace(0, duration, nsamps, endpoint=False)
        halfs = [np.s_[...]]
    else:
        raise ValueError(f'undefined value for part paramete: {part}')

    fig = go.Figure()
    for nn, half in enumerate(halfs):

        if hightlight_difference and mode == 'PSTH':
            # gray area between pair of contexts
            rgb = hex_to_rgb(Grey)  # tuple
            fill_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {CI_opacity})'
            x, y = squarefy(time[half],
                            eg_raster[:, contexts[0], half].mean(axis=0))
            _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                         line=dict(color='rgba(0,0,0,0)'),
                                         showlegend=False))
            x, y = squarefy(time[half],
                            eg_raster[:, contexts[1], half].mean(axis=0))
            _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                         line=dict(color='rgba(0,0,0,0)'),
                                         fill='tonexty', fillcolor=fill_color,
                                         showlegend=False))

        for cc, ctx_idx in enumerate(contexts):

            if prb_use_ctx_color:
                # the color associated with the context is also used
                # for the probe that follows
                color = color_palette[ctx_idx % len(color_palette)]
            elif not prb_use_ctx_color:
                # different sounds have different colors,
                # therefore the probe color can be different from
                # its preceding context.
                color = color_palette[
                    ctx_idx % len(color_palette)] if nn == 0 else \
                    color_palette[probe % len(color_palette)]

            else:
                raise ValueError(f'prb_use_ctx_color must be bool but is '
                                 f'{prb_use_ctx_color}')

            if nn == 0 and part == 'all':
                name = f'context {ctx_idx}'
            else:
                name = f'probe {probe} after context {ctx_idx}'

            if mode == 'PSTH':
                # for the second half prepend the last sample of the
                # first half to create a connector
                if part == 'all' and nn == 1:
                    y0 = np.mean(eg_raster[:, ctx_idx, halfs[0]], axis=0)[-1]
                else:
                    y0 = None

                _ = plot_PSTH(eg_raster[:, ctx_idx, half], time[half], y0=y0,
                              color=color, dash=dash, CI=CI,
                              CI_opacity=CI_opacity, name=name,
                              showlegend=showlegend, fig=fig)

            elif mode == 'raster':
                fig = plot_raster(eg_raster[:, ctx_idx, half], time[half],
                                  y0=nreps * cc, color=color, name=name,
                                  showlegend=showlegend, fig=fig)
            else:
                raise ValueError("undefined plot type, chose PSTH or raster")

    if part == 'all':
        x_range = [0 - duration / 2, duration / 2]
    elif part == 'probe':
        x_range = [0, duration]
    else:
        raise ValueError(f'undefined value for part paramete: {part}')

    _ = fig.update_xaxes(title_text='time from probe onset (s)',
                         title_standoff=0, range=x_range)
    _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot',
                      opacity=1)

    if mode == 'psth':
        _ = fig.update_yaxes(title_text='firing rate (z-score)',
                             title_standoff=0)
    elif mode == 'raster':
        _ = fig.update_yaxes(title_text='trials', title_standoff=0,
                             showticklabels=False, range=[0, nreps * 2])
    fig.update_layout(template='simple_white')

    return fig


def plot_raw_pair(cellid, contexts, probe, mode='PSTH', raster_fs=20, CI='std',
                  CI_opacity=0.5, hightlight_difference=False,
                  prb_use_ctx_color=False, part='all', pupil=False,
                  plot_pred=False, color_palette=FOURCOLOR, **kwargs):
    # Loads data, which can be responses or predictions.
    # In the case of predictions, it assumes no trial variability
    # and forces PSTH with no confidence interval.
    if 'modelname' in kwargs.keys() and 'batch' in kwargs.keys():
        fs = int(re.findall('\.fs\d*\.', kwargs['modelname'])[0][3:-1])
        if raster_fs != fs: print("enforcing model raster_fs")

        # note this is the model prediction, not the OG raster
        resp, pred, goodcells = load_cell_formated_resp_pred(cellid, part=part,
                                                             **kwargs)

        # we might want to plot the model prediction or the exact data
        # used to train the model
        if plot_pred:
            site_raster = pred
            dash = 'dot'
        else:
            site_raster = resp
            dash = 'solid'

        # force PSTH
        if mode != 'PSTH':
            print('can only plot psth for predictions, forcing...')
            mode = 'PSTH'
        if pupil is not False:
            raise NotImplementedError(
                'cannot make pupil distinction for model predictions')

    # in the case of responses, single trials can be plotted as a raster,
    # or a PSHT wiht confidence interval
    else:
        dash = 'solid'
        if mode == 'PSTH':
            fs = raster_fs
            # smoothing_window = 50
            smoothing_window = 0
        elif mode == 'raster':
            if raster_fs < 100:
                print(f'raster_fs={raster_fs} is too low for a good scatter. '
                      f'defaulting to 100hz')
                fs = 100
            else:
                fs = raster_fs
            smoothing_window = 0
        else:
            raise ValueError("undefined plot type, choose PSTH or raster")

        if "-PC-" in cellid:
            site_raster, goodcells = load_site_formated_PCs(cellid[:7],
                part=part, smoothing_window=smoothing_window, raster_fs=fs)
            goodcells = list(goodcells.keys())
        else:
            site_raster, goodcells = load_site_formated_raster(cellid[:7],
                part=part, smoothing_window=smoothing_window, raster_fs=fs)

        if pupil:
            pup_raster, _ = load_site_formated_raster(cellid[:7], part='all',
                                                      smoothing_window=0,
                                                      raster_fs=fs, pupil=True)

            pup_raster = np.mean(pup_raster, axis=-1, keepdims=True)
            pup_thresh = np.median(pup_raster, axis=0, keepdims=True)

            if pupil == 'big':
                pupil_mask = np.broadcast_to(pup_raster < pup_thresh,
                                             site_raster.shape)
            elif pupil == 'small':
                pupil_mask = np.broadcast_to(pup_raster >= pup_thresh,
                                             site_raster.shape)
            else:
                raise ValueError(
                    f"pupil parameter must be False, 'big' or 'small. "
                    f"receivede {pupil}")

            site_raster = np.ma.masked_where(pupil_mask, site_raster,
                                             copy=False)

    # the array loaded and formatted, calls the actual plotting with parameters
    # meaningful for the array source eg, we should not try to plot a
    # confidence interval with a deterministic model prediction
    fig = plot_raw_pair_array(site_raster, goodcells, cellid, contexts, probe,
                              raster_fs, part, prb_use_ctx_color,
                              hightlight_difference, CI, CI_opacity, mode,
                              dash, color_palette=color_palette)
    return fig


def plot_pupil_so_effects(cellid, contexts, probe, raster_fs=30,
                          error_opacity=0.2, ):
    prb_idx = probe - 1
    fs = raster_fs
    smoothing_window = 50

    site_raster, goodcells = load_site_formated_raster(cellid[:7],
                                                       part='probe',
                                                       smoothing_window=smoothing_window,
                                                       raster_fs=fs)

    # pupil size selected for the time average across context and probe response
    # this is for consistency with the same analysis done on scripts and jupyte notebooks
    pup_raster, _ = load_site_formated_raster(cellid[:7], part='all',
                                              smoothing_window=0, raster_fs=fs,
                                              pupil=True)

    # selects neuron, probe and pair of contexts to compare
    sel_raster = site_raster[:, goodcells.index(cellid), contexts, prb_idx, :]
    sel_pup = pup_raster[:, 0, contexts, prb_idx, :]

    # define pupil size threhsld independently for each context instance
    sel_pup = np.mean(sel_pup, axis=-1, keepdims=True)
    pup_thresh = np.median(sel_pup, axis=0, keepdims=True)

    # masks out trials with opossing pupil size
    small_mask = np.broadcast_to(sel_pup >= pup_thresh, sel_raster.shape)
    big_mask = np.broadcast_to(sel_pup < pup_thresh, sel_raster.shape)

    # defines common time values. No need to get
    _, _, nsamps = sel_raster.shape
    duration = nsamps / fs
    time = np.linspace(0, duration, nsamps, endpoint=False)

    fig = go.Figure()

    flipper = 1

    for pp, (pup_size, dash, pup_mask) in enumerate(
            zip(['big', 'small'], ['solid', 'dot'], [big_mask, small_mask])):

        pup_raster = np.ma.masked_where(pup_mask, sel_raster, copy=False)
        PSTHs = np.mean(pup_raster, axis=0)

        # delta FR
        DFR = PSTHs[0, :] - PSTHs[1, :]

        # confidence interval: standard error of the difference
        std = np.sqrt((sem(np.ma.compress_nd(pup_raster[:, 0, :], axis=0),
                           axis=0) ** 2) + (
                              sem(np.ma.compress_nd(pup_raster[:, 1, :],
                                                    axis=0), axis=0) ** 2))

        # check if big pupil DFR is negative, and sets flipper
        # both  big and small pupil traces are flipped together
        if pp == 0 and DFR.mean() < 0:
            flipper = -1
        DFR *= flipper

        x, y = squarefy(time, DFR)
        _, yerr = squarefy(time, std)

        # confidence interval shadow area
        rgb = hex_to_rgb('#000000')  # white -> tuple
        fill_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {error_opacity})'
        # transparent line in case its width is changed later outside this func
        line_color = 'rgba(0,0,0,0)'

        _ = fig.add_trace(
            go.Scatter(x=x, y=y + yerr, mode='lines', line_color=line_color,
                       line_width=0, showlegend=False))
        _ = fig.add_trace(
            go.Scatter(x=x, y=y - yerr, mode='lines', line_color=line_color,
                       line_width=0, fill='tonexty', fillcolor=fill_color,
                       showlegend=False))

        # meand Delta firing rate
        _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                     line=dict(color='black', dash=dash,
                                               width=3), name=pup_size,
                                     showlegend=True))

    _ = fig.update_xaxes(title_text='time from probe onset (s)',
                         title_standoff=0)
    _ = fig.update_yaxes(title_text='delta firing rate (z-score)',
                         title_standoff=0)
    fig.update_layout(template='simple_white')

    return fig


def plot_pop_modulation(cellid, modelname, batch, contexts, probe, **kwargs):
    fs = int(re.findall('\.fs\d*\.', modelname)[0][3:-1])
    _, mod_raster = get_population_influence(cellid, batch, modelname,
                                             **kwargs)

    toplot = mod_raster[0, 0, np.asarray(contexts), probe - 1, :]

    # asumes balanced data around zero
    _, nsamps = toplot.shape
    duration = nsamps / fs
    time = np.linspace(0 - duration / 2, duration / 2, nsamps, endpoint=False)

    colors = [Grey, Yellow, Red, Teal, Brown]
    fig = go.Figure()
    for cc, ctx_idx in enumerate(contexts):
        color = colors[ctx_idx % len(colors)]
        name = f'pop mod ctx{ctx_idx}_prb{probe}'

        x, y = squarefy(time, toplot[cc])
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', line_color=color, line_width=3,
                       line_dash='dash', name=name, showlegend=False))

    _ = fig.update_xaxes(title_text='time from probe onset (s)',
                         title_standoff=0,
                         range=[0 - duration / 2, duration / 2])
    _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot',
                      opacity=1)
    _ = fig.update_yaxes(title_text='pop modulation', title_standoff=0)

    return fig


def plot_pop_stategain(cellid, modelname, batch, orientation='v'):
    mean_pop_gain = get_population_weights(cellid, batch, modelname)
    if orientation == 'v':
        toplot = mean_pop_gain[:, None]
    elif orientation == 'h':
        toplot = mean_pop_gain[None, :]
    else:
        raise ValueError(
            f"unknown orientation value: {orientation}\nchoose 'v' or 'h'")

    img = go.Figure()
    # _ = img.add_trace(go.Heatmap(z=toplot, colorscale='BrBG', zmid=0))
    _ = img.add_trace(go.Heatmap(z=toplot, coloraxis="coloraxis"))
    # img = px.imshow(toplot, aspect='auto', color_continuous_scale='inferno')
    return img


def plot_mod_full(cellid, modelname, batch, contexts, probe, orientation='h',
                  **kwargs):
    mod_plot = plot_pop_modulation(cellid, modelname, batch, contexts, probe)

    if orientation == 'h':
        fig = make_subplots(1, 2, column_widths=[0.95, 0.05],
                            horizontal_spacing=0.01)
        # modulation
        fig.add_traces(mod_plot['data'], rows=[1] * len(mod_plot['data']),
                       cols=[1] * len(mod_plot['data']))

        # weigts
        weight_plot = plot_pop_stategain(cellid, modelname, batch,
                                         orientation='v')
        fig.add_traces(weight_plot['data'],
                       rows=[1] * len(weight_plot['data']),
                       cols=[2] * len(weight_plot['data']))

        fig.update_layout(coloraxis=dict(colorscale='inferno',
                                         colorbar=dict(orientation='v',
                                                       thickness=10, len=0.6,
                                                       title_text='weight',
                                                       title_side='right',
                                                       tickangle=0,
                                                       xanchor='left')))

    elif orientation == 'v':
        fig = make_subplots(2, 1, row_width=[0.05, 0.95],
                            vertical_spacing=0.01)
        # modulation
        fig.add_traces(mod_plot['data'], rows=[1] * len(mod_plot['data']),
                       cols=[1] * len(mod_plot['data']))

        # weigts
        weight_plot = plot_pop_stategain(cellid, modelname, batch,
                                         orientation='h')
        fig.add_traces(weight_plot['data'],
                       rows=[2] * len(weight_plot['data']),
                       cols=[1] * len(weight_plot['data']))

        fig.update_layout(coloraxis=dict(colorscale='inferno',
                                         colorbar=dict(orientation='v',
                                                       thickness=10, len=0.6,
                                                       title_text='weight',
                                                       title_side='right',
                                                       tickangle=0,
                                                       xanchor='left')))

    _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot',
                      opacity=1, row=1, col=1)
    fig.update_layout(xaxis2=dict(tickvals=[], ticktext=[]),
                      yaxis2=dict(tickvals=[], ticktext=[]))

    return fig


def plot_strf(cellid, modelname, batch):
    strf = get_strf(cellid, batch, modelname)
    img = px.imshow(strf, origin='lower', aspect='auto',
                    color_continuous_scale='inferno')

    return img


def plot_errors_over_time(cellid, modelname, batch, contexts, probe,
                          part='probe', grand_mean=False):
    err, diff_err = get_pred_err(cellid, batch, modelname, part)

    ctx, prb, tme = err.shape
    ctx_pairs = list(itt.combinations(range(ctx), 2))
    pidx = probe - 1  # probe names start from 1.

    fs = int(re.findall('\.fs\d*\.', modelname)[0][3:-1])
    duration = tme / fs
    if part == 'probe':
        time = np.linspace(0, duration, tme, endpoint=False)
    elif part == 'all':
        time = np.linspace(0 - duration / 2, duration / 2, tme, endpoint=False)

    fig = go.Figure()

    if grand_mean:
        # Take neuron mean accuracy for the cell. dont blame me,
        # Stephen made me do it!

        # first plot individual contex-probe error
        toplot = err.mean(axis=(0, 1))
        x, y = squarefy(time, toplot)
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', line_color='gray', line_width=2,
                       line_dash='dash', name='ctx_prb_mean', showlegend=True))

        # second add the difference error
        toplot = diff_err.mean(axis=(0, 1))
        x, y = squarefy(time, toplot)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='black',
                                 line_width=3, name='ctx-pair_prb_mean',
                                 showlegend=True))
    elif grand_mean is False:
        colors = [Grey, Yellow, Red, Teal, Brown]
        # first plot individual contex-probe error
        for cc, ctx_idx in enumerate(contexts):
            color = colors[ctx_idx % len(colors)]
            name = f'cxt{ctx_idx}_prb{probe}'
            toplot = err[ctx_idx, pidx, :]

            x, y = squarefy(time, toplot)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color,
                                     line_width=2, line_dash='dash', name=name,
                                     showlegend=True))

        # second add the difference error
        name = f'cxts{contexts}_prb{probe}'
        pair_idx = ctx_pairs.index(contexts)
        toplot = diff_err[pair_idx, pidx, :]

        x, y = squarefy(time, toplot)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='black',
                                 line_width=3, name=name, showlegend=True))
    else:
        raise ValueError(
            f"'grand_mean' should be bool but is {type(grand_mean)}")

    fig.update_layout(xaxis_title_text='time (ms)',
                      yaxis_title_text='sqr error')

    return fig


def plot_multiple_errors_over_time(cellid, modelnames, batch, contexts, probe,
                                   part='probe', style='mean', floor=None,
                                   nPCs=2):
    modelnicknames = {val: key for key, val in mns.modelnames.items()}
    colors = [Grey, Yellow, Red, Teal, Brown]

    if floor is not None:
        err_floor, diff_err_floor = get_pred_err(cellid, batch, floor, part)
        modelnames.pop(modelnames.index(floor))

    fig = go.Figure()
    for mm, modelname in enumerate(modelnames):
        err, diff_err = get_pred_err(cellid, batch, modelname, part)

        if floor is not None:
            err = err - err_floor
            diff_err = diff_err - diff_err_floor

        ctx, prb, tme = err.shape
        ctx_pairs = list(itt.combinations(range(ctx), 2))
        pidx = probe - 1  # probe names start from 1.

        fs = int(re.findall('\.fs\d*\.', modelname)[0][3:-1])
        duration = tme / fs
        if part == 'probe':
            time = np.linspace(0, duration, tme, endpoint=False)
        elif part == 'all':
            time = np.linspace(0 - duration / 2, duration / 2, tme,
                               endpoint=False)

        color = colors[mm % len(colors)]
        if style == 'mean':
            name = f'{modelnicknames[modelname]}'
            toplot = diff_err.mean(axis=(0, 1))
            x, y = squarefy(time, toplot)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color,
                                     line_width=3, name=name, showlegend=True))
        elif style == 'instance':
            name = f'{modelnicknames[modelname]}_C{contexts}_P{probe}'
            pair_idx = ctx_pairs.index(tuple(contexts))
            toplot = diff_err[pair_idx, pidx, :]

            x, y = squarefy(time, toplot)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color,
                                     line_width=3, name=name, showlegend=True))

        elif style == 'PCA':

            pca = PCA(n_components=50)
            tofit = diff_err.reshape((-1, diff_err.shape[-1]))
            toplot = pca.fit_transform(tofit.T).T  # PC by Time

            dashings = ['solid', 'dash', 'dot']
            for pc in range(nPCs):
                name = f'{modelnicknames[modelname]}_PC{pc}'
                x, y = squarefy(time, toplot[pc, :])
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                         line=dict(color=color, width=3,
                                                   dash=dashings[
                                                       pc % len(dashings)]),
                                         name=name, showlegend=True))

                # inset for variance explained
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                    y=np.cumsum(pca.explained_variance_ratio_),
                    mode='lines+markers', line_color=color, marker_color=color,
                    xaxis='x2', yaxis='y2', showlegend=False))

            fig.update_layout(
                xaxis2=dict(domain=[0.7, 0.95], anchor='y2', title_text='PC#'),
                yaxis2=dict(domain=[0.7, 0.95], anchor='x2',
                            title_text='var explained'))

        else:
            raise ValueError(
                f"{style} unrecognized 'style' value. Use 'instance', 'mean' or 'PCA'")

        fig.update_layout(xaxis_title_text='time (ms)',
                          yaxis_title_text='sqr error')

    return fig


def plot_model_prediction_comparison(cellid, batch, independent_models,
                                     dependent_model, contexts, probe,
                                     part='probe', grand_mean=False):
    modelnicknames = {val: key for key, val in mns.modelnames.items()}
    colors = [Grey, Yellow, Red, Teal, Brown]

    rasters, aggs = model_independence_comparison(cellid, batch,
                                                  independent_models,
                                                  dependent_model, part=part)

    ctx, prb, tme = rasters[dependent_model].shape
    ctx_pairs = list(itt.combinations(range(ctx), 2))
    pidx = probe - 1  # probe names start from 1.

    # asumes the same sampling rate for all the models. Reasonable asumptions.
    fs = int(re.findall('\.fs\d*\.', dependent_model)[0][3:-1])
    duration = tme / fs
    if part == 'probe':
        time = np.linspace(0, duration, tme, endpoint=False)
    elif part == 'all':
        time = np.linspace(0 - duration / 2, duration / 2, tme, endpoint=False)

    fig = make_subplots(1, 3)

    # model predicitions
    for cc, ctx_idx in enumerate(contexts):
        # thin textured line for independent model predictions

        toplots = [rasters[independent_models[0]][ctx_idx, pidx, :],
                   rasters[independent_models[1]][ctx_idx, pidx, :], (rasters[
                                                                          independent_models[
                                                                              0]] +
                                                                      rasters[
                                                                          independent_models[
                                                                              1]])[
                                                                     ctx_idx,
                                                                     pidx, :],
                   rasters[dependent_model][ctx_idx, pidx, :]]
        names = [f"{modelnicknames[independent_models[0]]}",
                 f"{modelnicknames[independent_models[1]]}", 'model_sum',
                 f"{modelnicknames[dependent_model]}"]
        dashings = ['dash', 'dot', 'dashdot', 'solid']
        widths = [1, 1, 3, 3.5]

        color = colors[ctx_idx % len(colors)]
        for toplot, name, dashing, width in zip(toplots, names, dashings,
                                                widths):
            x, y = squarefy(time, toplot)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color,
                                     line_dash=dashing, line_width=width,
                                     name=name, showlegend=True), row=1,
                          col=cc + 1)

    # all context probes for the neuron
    to_concat = list()
    for pred_source in ['dependent', 'sum']:
        df = pd.DataFrame(index=range(ctx), columns=range(1, prb + 1),
                          data=aggs[pred_source])
        df['model'] = pred_source
        to_concat.append(df)

    df = pd.concat(to_concat, axis=0)
    df.index.name = 'context'
    df.columns.name = 'probe'
    df.reset_index(inplace=True)

    toplot = df.melt(id_vars=['context', 'model']).pivot_table(
        index=['context', 'probe'], columns='model', values='value',
        aggfunc='first').reset_index()
    scatter = px.scatter(toplot, x='dependent', y='sum',
                         hover_data=['context', 'probe'],
                         color_discrete_sequence=['black'])

    fig.add_traces(scatter['data'], rows=[1] * len(scatter['data']),
                   cols=[3] * len(scatter['data']))

    # highlighs the selected  values
    selected = toplot.query(f"context in {contexts} and probe == {probe}")
    fig.add_trace(
        go.Scatter(x=selected['dependent'], y=selected['sum'], mode='markers',
                   marker_color=[colors[int(cc) % len(colors)] for cc in
                                 selected['context']], marker_size=10), row=1,
        col=3)

    # unit line
    plot_range = [toplot.loc[:, ['dependent', 'sum']].values.min(),
                  toplot.loc[:, ['dependent', 'sum']].values.max()]
    fig.add_trace(go.Scatter(x=plot_range, y=plot_range, mode='lines',
                             line_color='black', line_dash='dash'), row=1,
                  col=3)

    fig.update_xaxes(title_text=modelnicknames[dependent_model], col=3, row=1)
    fig.update_yaxes(scaleratio=1,
                     title_text=f'{modelnicknames[independent_models[0]]} + {modelnicknames[dependent_model][0]}',
                     col=3, row=1)

    return fig


def plot_time_ser_quant(cellid, contexts, probe, multiple_comparisons_axis,
                        cluster_threshold, alpha=0.05, source='real',
                        secondary_y=False, deltaFR=False, ignore_quant=False,
                        meta={}):
    """
    plot showing the quantification of time series differences (PSTHs) between
    context effects. It shows the difference metric (t-score), its threshold
    for cluster deffinition, the t-score sume for each cluster, and the
    threshold for cluster significance based on the spermutation distribution.
    It also displays the area of time bins in clusters that are significant,
    alongside the center of mass of this significant area.
    """
    raster_meta = {'montecarlo': 11000, 'raster_fs': 30, 'reliability': 0.1,
                   'smoothing_window': 0, 'stim_type': 'permutations',
                   'zscore': True}

    raster_meta.update(meta)
    montecarlo = raster_meta.pop('montecarlo')

    if "-PC-" in cellid:
        load_fn = 'PCA'
    else:
        load_fn = 'SC'

    if tstat_cluster_mass.check_call_in_cache(cellid[:7],
                                              cluster_threshold=float(
                                                  cluster_threshold),
                                              montecarlo=montecarlo,
                                              raster_meta=raster_meta,
                                              load_fn=load_fn):
        dprime, pval_quantiles, goodcells, shuffled_eg = tstat_cluster_mass(
            cellid[:7], cluster_threshold=float(cluster_threshold),
            montecarlo=montecarlo, raster_meta=raster_meta, load_fn=load_fn)
    else:
        raise ValueError(
            f'{cellid[:7]}, {tstat_cluster_mass}, {cluster_threshold} not yet in cache')

    # uses the absolute delta FR to calculate integral
    if deltaFR:
        dfr = pairwise_delta_FR(cellid[:7], raster_meta=raster_meta,
                                load_fn=load_fn)

    if source == 'real':
        pvalue = pval_quantiles['pvalue']
    elif source == 'shuffled_eg':
        dprime = shuffled_eg['dprime']
        pvalue = shuffled_eg['pvalue']

    significance = _significance(pvalue,
                                 multiple_comparisons_axis=multiple_comparisons_axis,
                                 alpha=alpha)

    # new alphas corrected by multiple comparisons
    if multiple_comparisons_axis is None:
        mult_comp = 'none'
    elif len(multiple_comparisons_axis) == 2:
        # asumes correction across context and probes
        mult_comp = 'bf_cp'
    elif len(multiple_comparisons_axis) == 3:
        # asumes correction across neurons, context and probes
        mult_comp = 'bf_ncp'
    else:
        raise ValueError(
            'I dont know what to do with so many multiple_comparisons_axis')

    if type(goodcells) is dict:
        goodcells = list(goodcells.keys())
    cell_idx = goodcells.index(cellid)
    pair_idx = [f'{t0}_{t1}' for t0, t1 in
                itt.combinations(range(dprime.shape[2] + 1), 2)].index(
        f'{contexts[0]}_{contexts[1]}')
    prb_idx = probe - 1

    # figures out if flip is neede
    DP = dprime[cell_idx, pair_idx, prb_idx, :]

    if np.sum(DP) < 0:
        flip = -1
    else:
        flip = 1

    DP *= flip

    if deltaFR:
        dfr = dfr[cell_idx, pair_idx, prb_idx, :]
        dfr *= flip

    if mult_comp in pval_quantiles.keys():
        CI = pval_quantiles[mult_comp][cell_idx, pair_idx, prb_idx, 0]
    else:
        raise ValueError(f'undefined quantiles for {mult_comp}')

    CT = pval_quantiles['clusters'][cell_idx, pair_idx, prb_idx, :] * flip

    if 't-threshold' in pval_quantiles:
        # dinamically defined threshold for t test. depends on degrees of fredom i.e. reps
        CTT = pval_quantiles['t-threshold']
        print(
            f'using t-score threshold for sample-alpha {cluster_threshold} -> t = {CTT}')
    else:
        CTT = cluster_threshold

    SIG = significance[cell_idx, pair_idx, prb_idx, :]

    signif_mask = SIG > 0
    t = np.linspace(0, DP.shape[-1] / raster_meta['raster_fs'], DP.shape[-1],
                    endpoint=False)

    # calculates integral, center of mass and last bin
    if not ignore_quant:
        if deltaFR:
            to_quantify = dfr
        else:
            to_quantify = DP

        integral = np.sum(np.abs(to_quantify[signif_mask])) * np.mean(
            np.diff(t))
        print(f"integral: {integral * 1000:.2f} t-score*ms")

        mass_center = np.sum(
            np.abs(to_quantify[signif_mask]) * t[signif_mask]) / np.sum(
            np.abs(to_quantify[signif_mask]))
        if np.isnan(mass_center): mass_center = 0
        print(f'center of mass: {mass_center * 1000:.2f} ms')

        if np.any(signif_mask):
            dt = np.mean(np.diff(t))
            mt = t + dt
            last_bin = np.max(mt[signif_mask])
        else:
            last_bin = 0

        print(f'last bin: {last_bin * 1000:.2f} ms')

    fig = make_subplots(specs=[[{"secondary_y": secondary_y}]])

    # holds tracese for the left and potentially right y axes in lists for easy addition to figure and extraction for more
    # complex figures. e.g paper
    main_traces = list()
    secondary_traces = list()

    # plots main metric e.g. t-stat and cluster threshold on primary axis
    # if using delta firing rate, plot it as a solid line, and the T-score as a dotted line
    if deltaFR:
        tt, mmdd = squarefy(t, dfr)
        linename = "delta FR"
    else:
        tt, mmdd = squarefy(t, DP)
        linename = 'T-statistic'

    main_traces.append(
        go.Scatter(x=tt, y=mmdd, mode='lines', line_color='black',
                   line_width=3, name=linename))
    main_traces.append(go.Scatter(x=tt[[0, -1]], y=[CTT] * 2, mode='lines',
                                  line=dict(color='Black', dash='dash',
                                            width=2), name='T-stat thresold'))

    if deltaFR:
        # here adds the T-score values as a dotted line
        tt, ttss = squarefy(t, DP)
        main_traces.append(go.Scatter(x=tt, y=ttss, mode='lines',
                                      line=dict(color='black', dash='dot',
                                                width=3), name='T-statistic'))

    ## significant area under the curve
    if not ignore_quant:
        _, smm = squarefy(t, signif_mask)
        wmmdd = np.where(smm, mmdd,
                         0)  # little hack to add gaps into the area, set y value to zero where no significance
        rgb = hex_to_rgb(AMPCOLOR)
        rgba = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.5)'

        main_traces.append(
            go.Scatter(x=tt, y=wmmdd, mode='none', fill='tozeroy',
                       fillcolor=rgba, name='integral'))

        ## Center of mass indication: line fom zero to the time series value
        # at that time point.
        if not np.isnan(mass_center):
            ytop = DP[np.abs(t - mass_center).argmin()]
            main_traces.append(
                go.Scatter(x=[mass_center] * 2, y=[0, ytop], mode='lines',
                           line=dict(color=DURCOLOR, width=4),
                           name='center of mass'))

        ## adds star at the last time bin position
        if last_bin > 0:
            main_traces.append(go.Scatter(x=[last_bin], y=[0], mode='markers',
                                          marker=dict(symbol='star',
                                                      color=DURCOLOR, size=15),
                                          name='last bin'))

    for trace in main_traces:
        # forces main traces to be on the primary y ax
        fig.add_trace(trace,
                      secondary_y=False)

    # Cluster and corrected confidence interval of the shuffled clusters,
    # this can be on a secondary y axis.
    tt, mmcc = squarefy(t, CT)
    secondary_traces.append(go.Scatter(x=tt, y=mmcc, mode='lines',
                                       line=dict(color=Green, dash='solid',
                                                 width=3), name='cluster sum'))
    secondary_traces.append(go.Scatter(x=tt[[0, -1]], y=[CI] * 2, mode='lines',
                                       line=dict(color=Green, dash='dash',
                                                 width=2),
                                       name='cluster threshold'))

    for trace in secondary_traces:
        fig.add_trace(trace, secondary_y=secondary_y)

    # formats axis, legend and so on.
    _ = fig.update_xaxes(
        title=dict(text='time from probe onset (s)', standoff=0))

    _ = fig.update_yaxes(title=dict(text="difference (t-score)", standoff=0))
    if secondary_y:
        _ = fig.update_yaxes(
            title=dict(text="cluster sum (t-score)", standoff=0),
            secondary_y=secondary_y)

    return fig, main_traces, secondary_traces


def plot_simple_quant(cellid, contexts, probe, multiple_comparisons_axis,
                      cluster_threshold, alpha=0.05, meta={}):
    """
    plot shoing the quantification of time series differences (PSTHs) between
    context effects. It shows the difference metric (t-score), its threshold
    for cluster deffinition, the t-score sume for each cluster, and the
    threshold for cluster significance based on the spermutation distribution.
    It also displays the are of time bins in clusters that are significant,
    alongside the center of mass of this significant area.
    """
    raster_meta = {'montecarlo': 11000, 'raster_fs': 30, 'reliability': 0.1,
                   'smoothing_window': 0, 'stim_type': 'permutations',
                   'zscore': True}

    raster_meta.update(meta)
    montecarlo = raster_meta.pop('montecarlo')

    if "-PC-" in cellid:
        load_fn = 'PCA'
    else:
        load_fn = 'SC'

    if tstat_cluster_mass.check_call_in_cache(
            cellid[:7], cluster_threshold=float(cluster_threshold),
            montecarlo=montecarlo, raster_meta=raster_meta,load_fn=load_fn
    ):

        _, pval_quantiles, goodcells, _ = tstat_cluster_mass(
            cellid[:7], cluster_threshold=float(cluster_threshold),
            montecarlo=montecarlo, raster_meta=raster_meta, load_fn=load_fn
        )
    else:
        raise ValueError(
            f'{cellid[:7]}, {tstat_cluster_mass}, {cluster_threshold} '
            f'not yet in cache'
        )

    # uses the absolute delta FR to calculate integral
    DFR = pairwise_delta_FR(cellid[:7], raster_meta=raster_meta,
                            load_fn=load_fn)

    significance = _significance(
        pval_quantiles['pvalue'],alpha=alpha,
        multiple_comparisons_axis=multiple_comparisons_axis,
    )

    if type(goodcells) is dict:
        goodcells = list(goodcells.keys())
    cell_idx = goodcells.index(cellid)
    pair_idx = [f'{t0}_{t1}' for t0, t1 in
                itt.combinations(range(DFR.shape[2] + 1), 2)].index(
        f'{contexts[0]}_{contexts[1]}')
    prb_idx = probe - 1

    # figures out if flip is neede
    DFR = DFR[cell_idx, pair_idx, prb_idx, :]
    DFR = np.abs(DFR)

    SIG = significance[cell_idx, pair_idx, prb_idx, :]

    signif_mask = SIG > 0
    t = np.linspace(0, DFR.shape[-1] / raster_meta['raster_fs'], DFR.shape[-1],
                    endpoint=False)

    # calculates center of mass and integral
    integral = np.sum(np.abs(DFR[signif_mask])) * np.mean(np.diff(t))
    print(f"integral: {integral:.3f} Z-score*s")

    mass_center = np.sum(np.abs(DFR[signif_mask]) * t[signif_mask]) / np.sum(
        np.abs(DFR[signif_mask]))
    if np.isnan(mass_center): mass_center = 0
    print(f'center of mass: {mass_center * 1000:.2f} ms')

    if np.any(signif_mask):
        dt = np.mean(np.diff(t))
        mt = t + dt
        last_bin = np.max(mt[signif_mask])
    else:
        last_bin = 0

    print(f'last bin: {last_bin * 1000:.2f} ms')

    fig = go.Figure()
    main_traces = list()

    # plots delta firing rate
    tt, mmdd = squarefy(t, DFR)

    main_traces.append(go.Scatter(x=tt, y=mmdd, mode='lines',
                                  line=dict(color='black', width=3),
                                  name="delta FR"))

    ## significant area under the curve
    _, smm = squarefy(t, signif_mask)

    # Little hack to add gaps into the area,
    # set y value to zero where no significance.
    wmmdd = np.where(smm, mmdd,
                     0)
    rgb = hex_to_rgb(AMPCOLOR)
    rgba = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.5)'

    main_traces.append(
        go.Scatter(x=tt, y=wmmdd, mode='none', fill='tozeroy', fillcolor=rgba,
                   name='integral'))

    ## center of mass indication:
    # line fom zero to the time series value at that time point
    if not np.isnan(mass_center):
        ytop = DFR[np.abs(t - mass_center).argmin()]
        main_traces.append(
            go.Scatter(x=[mass_center] * 2, y=[0, ytop], mode='lines',
                       line=dict(color=DURCOLOR, width=4),
                       name='center of mass'))

    ## adds star at the last time bin position
    if last_bin > 0:
        main_traces.append(go.Scatter(x=[last_bin], y=[0], mode='markers',
                                      marker=dict(symbol='star',
                                                  color=DURCOLOR, size=15),
                                      name='last bin'))

    fig.add_traces(main_traces)  # forces main traces to be on the primary y ax

    # formats axis, legend and so on.
    _ = fig.update_xaxes(
        title=dict(text='time from probe onset (s)', standoff=0))

    _ = fig.update_yaxes(title=dict(text="delta Firing Rate", standoff=0))

    return fig


def plot_tiling(picked_id, df, zmax=None, time_metric='last_bin',
                show_coloraxis=True, orientation='h', cscales=None):
    # turns long format data into an array with dimension Probe * context_pair
    if len(picked_id) == 7:
        # site case, get max projection across neurons
        to_pivot = df.query(
            f"site == '{picked_id}' "
            f"and metric in ['integral', '{time_metric}']"
        ).groupby(
            ['metric', 'probe', 'context_pair']
        ).agg(value=('value', 'max'))
    else:
        # neuron case, just select data
        to_pivot = df.query(
            f"id == '{picked_id}' and metric in ['integral', '{time_metric}']")
    val_df = to_pivot.pivot_table(index=['metric', 'probe'],
                                  columns=['context_pair'], values='value')

    if cscales is None:
        cscales = {'integral': pc.make_colorscale(['#FFFFFF', Green]),
                   time_metric: pc.make_colorscale(['#FFFFFF', Purple])}

    max_vals = dict()

    # normalizes,saves max values and get colors for each metric
    color_df = val_df.copy()
    for metric in color_df.index.levels[0]:
        if zmax is None:
            max_vals[metric] = val_df.loc[metric].values.max()
        else:
            assert type(zmax) is dict
            max_vals = zmax

        colors = pc.sample_colorscale(cscales[metric], (
                val_df.loc[metric] / max_vals[metric]).values.flatten())
        color_df.loc[metric] = np.asarray(colors).reshape(
            color_df.loc[metric].shape)

    # general shapes of the upper and lower triangles
    # to be passed to Scatter x and y
    xu, yu = np.array([0, 0, 1, 0]), np.array([0, 1, 1, 0])
    xl, yl = np.array([0, 1, 1, 0]), np.array([0, 0, 1, 0])

    amp_color = color_df.loc[('integral'), :].values
    dur_color = color_df.loc[(time_metric), :].values

    amplitudes = val_df.loc[('integral'), :].values
    durations = val_df.loc[(time_metric), :].values

    if orientation == 'h':
        pass
    elif orientation == 'v':
        amp_color = amp_color.T
        dur_color = dur_color.T
        amplitudes = amplitudes.T
        durations = durations.T
    else:
        raise ValueError(f"invalid orientation value {orientation}")

    fig = go.Figure()

    for nn, (p, c) in enumerate(np.ndindex(amp_color.shape)):
        # note the use of transparent markers to define the colorbars internally
        # amplitud uppe half
        if amplitudes[p, c] == 0:
            continue

        _ = fig.add_scatter(x=xu + c, y=yu + p, mode='lines+markers',
                            line_width=1, line_color='#222222', fill='toself',
                            fillcolor=amp_color[p, c],
                            marker=dict(color=(amplitudes[p, c],) * len(xu),
                                        coloraxis='coloraxis', opacity=0,
                                        cmin=0, cmax=max_vals['integral'], ),
                            showlegend=False)

        # duration lower half
        _ = fig.add_scatter(x=xl + c, y=yl + p, mode='lines+markers',
                            line_width=1, line_color='#222222', fill='toself',
                            fillcolor=dur_color[p, c],
                            marker=dict(color=(durations[p, c],) * len(xl),
                                        coloraxis='coloraxis2', opacity=0,
                                        cmin=0, cmax=max_vals[time_metric], ),
                            showlegend=False)

    # adds encasing margin
    h, w = amplitudes.shape
    x = [0, w, w, 0, 0]
    y = [0, 0, h, h, 0]
    fig.add_trace(
        go.Scatter(x=x, y=y, mode='lines', line=dict(color='black', width=1),
                   showlegend=False))

    # label handling
    context_pairs = [f"{int(pp.split('_')[0])}_{int(pp.split('_')[1])}" for pp
                     in val_df.columns.to_list()]
    probes = val_df.index.get_level_values(1).unique().to_list()

    if orientation == 'h':
        xrange = [0, len(context_pairs)]
        xtitle = 'context pairs'
        xtickvals = np.arange(len(context_pairs)) + 0.5
        xticktext = context_pairs

        yrange = [0, len(probes)]
        ytitle = 'probes'
        ytickvals = np.arange(len(probes)) + 0.5
        yticktext = probes

    elif orientation == 'v':
        xrange = [0, len(probes)]
        xtitle = 'probes'
        xtickvals = np.arange(len(probes)) + 0.5,
        xticktext = probes

        yrange = [0, len(context_pairs)]
        ytitle = 'context pairs'
        ytickvals = np.arange(len(context_pairs)) + 0.5,
        yticktext = context_pairs
    else:
        raise ValueError(f"invalid orientation value {orientation}")

    fig.update_xaxes(dict(scaleanchor='y', constrain='domain', range=xrange,
                          fixedrange=True, title_text=xtitle, tickmode='array',
                          tickvals=xtickvals, ticktext=xticktext,
                          showticklabels=True))
    fig.update_yaxes(dict(constrain='domain', range=yrange, fixedrange=True,
                          title_text=f'{picked_id}<br>{ytitle}',
                          tickmode='array', tickvals=ytickvals,
                          ticktext=yticktext, showticklabels=True)

                     )
    # set the positions of the colorbars
    if show_coloraxis:
        fig.update_layout(coloraxis=dict(colorscale=cscales['integral'],
                                         colorbar=dict(thickness=10, len=0.6,
                                                       title_text='integral',
                                                       title_side='right',
                                                       tickangle=-90,
                                                       xanchor='left', x=1)),
                          coloraxis2=dict(colorscale=cscales[time_metric],
                                          colorbar=dict(thickness=10, len=0.6,
                                                        title_text=time_metric,
                                                        title_side='right',
                                                        tickangle=-90,
                                                        xanchor='left',
                                                        x=1.1)))
    else:
        fig.update_layout(coloraxis=dict(showscale=False),
                          coloraxis2=dict(showscale=False))

    fig.update_layout(template='simple_white')

    return fig


def plot_model_fitness(cellids, modelnames, nicknames=None, stat='r_test',
                       mode='bars'):
    """
    parses data from CPN experiments (batchs) and models into a figure
    displaying r-test
    """
    DF = batch_comp(batch=[326, 327], modelnames=modelnames, cellids=cellids,
                    stat=stat)
    if nicknames is not None:
        DF.columns = nicknames
        mod_cols = nicknames
    else:
        mod_cols = modelnames

    DF.reset_index(inplace=True)

    A1df = get_batch_ids(326)
    A1df['region'] = 'A1'
    PEGdf = get_batch_ids(327)
    PEGdf['region'] = 'PEG'
    regdf = pd.concat([A1df, PEGdf])

    wide = pd.merge(DF, regdf, on='cellid').rename(
        columns={'cellid': 'id', 'siteid': 'site'})

    if mode == 'bars':
        long = pd.melt(wide, id_vars=['id', 'site', 'region'],
                       value_vars=mod_cols, var_name='model', value_name=stat)
        fig = px.box(long, x='model', y=stat, color='region', points='all')

    return fig


def plot_cell_coverage(fnDF, cellid, zero_nan=True):
    celldf = fnDF.query(f"id == '{cellid}'")
    z = celldf.pivot(index='context_pair', columns='probe', values='value')

    if zero_nan:
        z[z == 0] = np.nan

    heatmap = go.Figure(
        go.Heatmap(z=z, zmid=0, coloraxis='coloraxis', connectgaps=False))

    # adds text if column present in dataframe and has something other than NaN
    # %%
    if "text" in celldf.columns:
        if np.any(~celldf["text"].isna()):
            # here text has to be passed as a DF with context and probes
            # as indices and columns
            text_df = celldf.pivot(index='context_pair', columns='probe',
                                   values='text')
            heatmap.update_traces(text=text_df, texttemplate='%{text}')

    try:
        thismax = np.nanmax(z)
    except:
        thismax = 0

    return heatmap, thismax


def plot_site_coverages(fnDF, cells_toplot='all', has_neg=False, rows=None,
                        cols=None):
    if cells_toplot == 'all':
        cells_toplot = fnDF.id.unique()
    else:
        pass

    print(cells_toplot)

    if has_neg:
        zero_nan = False
        colorscale = 'BrBg'
        cmid = 0
    else:
        zero_nan = True
        colorscale = 'inferno'
        cmid = None

    if rows == None and cols == None:
        rows, cols = square_rows_cols(len(cells_toplot))

    max_vals = list()
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes='all',
                        shared_yaxes='all', horizontal_spacing=0.01,
                        vertical_spacing=0.05, subplot_titles=[
            cid[8:] if cid not in ["Union", "PC1", "dense"] else cid for cid in
            cells_toplot], )

    # individual neuron examples
    for cc, cell_eg in enumerate(cells_toplot):
        hmap, maxval = plot_cell_coverage(fnDF, cell_eg, zero_nan=zero_nan)
        max_vals.append(maxval)
        row, col = int(np.floor(cc / cols)) + 1, (cc % cols) + 1
        hmap = hmap['data']
        fig.add_traces(hmap, rows=[row] * len(hmap), cols=[col] * len(hmap))

    ##### formating #######
    # reduces size of subplot titles
    fig.update_annotations(font_size=10)

    # ensures spines, and no ticks or tick labels
    fig.update_xaxes(scaleanchor='y', constrain='domain', showticklabels=False,
                     ticks='', showline=True, mirror=True, )

    fig.update_yaxes(constrain='domain', showticklabels=False, ticks='',
                     showline=True, mirror=True, )

    # labels on top left panel
    df = fnDF.query(f"id == '{cells_toplot[0]}'").pivot(index='context_pair',
                                                        columns='probe',
                                                        values='value')
    ctx_prs = [f"{int(pp.split('_')[0])}_{int(pp.split('_')[1])}" for pp in
               df.index.to_list()]
    prbs = df.columns.tolist()

    # probes, x axis
    fig.update_xaxes(title=dict(text='probe', standoff=0, font_size=10),
                     showticklabels=True, tickmode='array',
                     tickvals=list(range(len(prbs))), ticktext=prbs,
                     tickfont_size=9, col=1, row=1)
    # context pairs, y axis
    fig.update_yaxes(title=dict(text='context_pair', standoff=0, font_size=10),
                     showticklabels=True, tickmode='array',
                     tickvals=list(range(len(ctx_prs))), ticktext=ctx_prs,
                     tickangle=0, tickfont_size=9, col=1, row=1)

    w, h = 4 * 96, 4.5 * 96
    fig.update_layout(
        template="simple_white", width=w, height=h,
        margin=dict(l=10, r=10, t=30, b=10),
        coloraxis=dict(
            showscale=True,
            colorscale=colorscale,
            cmid=cmid,
            colorbar=dict(
                orientation='v',
                thicknessmode='fraction',
                thickness=0.02,
                lenmode='fraction', len=1,
                tickangle=-50,
                tickfont_size=9,
                title=dict(
                    text='Amplitude (delta Z-Score)',
                    side='right',
                    font_size=10
                )
            )
        )
    )

    return fig


def plot_ctx_clusters(fnArr: np.array, idxr: tuple[np.array],
                      trial_mode: str = 'scatter', jitter: float = 0.0,
                      n_std: float = 1.96,
                      color_palette: list[str] = FOURCOLOR,
                      showlegend: bool = False) -> go.Figure:
    # select consisten marker sybols and line dashings depending on probe
    all_symbols = ['square', 'diamond', 'circle',
                   'star-triangle-up']  # just 2 probes for clarity
    all_dashings = ['dot', 'dash', 'solid', 'dashdot']
    # symbols = [all_symbols[pp] for pp in idxr[3].squeeze()]
    # dashings = [all_dashings[pp] for pp in idxr[3].squeeze()]

    # select consistent colors dependent on context
    colors = [color_palette[ii % 10] for ii in idxr[2].squeeze()]

    # slices the array into the desired data view
    cells = idxr[1].squeeze()
    contexts = idxr[2].squeeze()
    probes = idxr[3].squeeze()
    time = idxr[4].squeeze()
    slcArr = fnArr[idxr]
    rep, chn, ctx, prb, tme = slcArr.shape
    assert chn == 2  # can only plot two neurons in the plane
    assert tme == 1  # can only plot one time point

    # handle single probe examples
    if probes.ndim == 0:
        probes = probes[None]
    symbols = [all_symbols[pp] for pp in probes]
    dashings = [all_dashings[pp] for pp in probes]

    fig = go.Figure()
    # single trials and means with different  marker sizes and opacioty
    arrs = [slcArr, slcArr.mean(axis=0, keepdims=True)]
    markersizes = [5, 10]
    opacities = [0.8, 1]

    # save ellipses data to define range of plots
    all_ellipses = list()

    for pp, prbidx in enumerate(probes):

        symbol = symbols[pp]
        dashing = dashings[pp]

        for cc, ctxidx in enumerate(contexts):
            for arr, ms, op in zip(arrs, markersizes, opacities):
                x = arr[:, 0, cc, pp, 0]
                y = arr[:, 1, cc, pp, 0]
                nreps = x.shape[0]

                name = f"ctx-{ctxidx} prb-{prbidx + 1}"
                if nreps > 1:  # single trial handling
                    # add some jitter to single trials
                    if trial_mode == "scatter":
                        if jitter != 0:
                            jitarr = np.random.uniform(-jitter, jitter,
                                                       (nreps, 2))
                            x = x + jitarr[:, 0]
                            y = y + jitarr[:, 1]

                        _ = fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                            marker=dict(symbol=symbol, color=colors[cc],
                                        size=ms, opacity=op),
                            showlegend=False), )

                    elif trial_mode == "ellipse":
                        trc = confidence_ellipse(x=x, y=y, n_std=n_std,
                                                 line=dict(color=colors[cc],
                                                           dash=dashing,
                                                           width=3), name=name,
                                                 showlegend=showlegend)
                        all_ellipses.append(
                            np.stack((trc['x'], trc['y']), axis=1))
                        _ = fig.add_trace(trc)
                    else:
                        raise ValueError(
                            f"trial_mode must be 'scatter' or 'ellipse' but is"
                            f" {trial_mode}")

                else:  # average
                    _ = fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                                 marker=dict(symbol=symbol,
                                                             size=ms,
                                                             opacity=op,
                                                             color=colors[cc],
                                                             line=dict(
                                                                 color='black',
                                                                 width=1)),
                                                 name=name,
                                                 showlegend=showlegend), )

        # average across all contexts. even those not being displayed
        # Creates a new indexer that selects the desired neurons and probes
        avgidxr = np.ix_(np.arange(fnArr.shape[0]), cells,
                         np.arange(fnArr.shape[2]), [prbidx], [time])
        grandmeam = fnArr[avgidxr].mean(axis=(0, 2)).squeeze()
        _ = fig.add_trace(
            go.Scatter(x=[grandmeam[0]], y=[grandmeam[1]], mode='markers',
                       marker=dict(color='black', symbol=symbol, size=12),
                       name=f"probe {prbidx} average", showlegend=showlegend))

    # diagonals
    if trial_mode == 'scatter':
        dd = np.asarray([np.min(slcArr), np.max(slcArr)])
    elif trial_mode == 'ellipse':
        all_ellipses = np.stack(all_ellipses, axis=0)
        dd = np.asarray([np.min(all_ellipses), np.max(all_ellipses)])
    _ = fig.add_trace(go.Scatter(x=dd, y=dd, mode='lines',
                                 line=dict(color='black', dash='dot'),
                                 opacity=0.5, name='sign threshold',
                                 showlegend=False))

    fig.add_vline(0, line_color='black', line_dash='dash')
    fig.add_hline(0, line_color='black', line_dash='dash')

    # axis labels
    _ = fig.update_yaxes(scaleanchor='x', scaleratio=1,
                         title=dict(text='neuron 2 activity (AU)', standoff=0))
    _ = fig.update_xaxes(title=dict(text='neuron 1 activity (AU)', standoff=0))

    return fig


def plot_eg_diag(fnArraList: list, idxr: tuple[np.array],
                 trial_mode: str = 'scatter', jitter: float = 0.0,
                 n_std: float = 1.96, orientation: str = 'h') -> go.Figure:
    if orientation == 'h':
        fig = make_subplots(1, len(fnArraList), shared_xaxes='all',
                            shared_yaxes='all', horizontal_spacing=0.01,
                            vertical_spacing=0.01)
    elif orientation == 'v':
        fig = make_subplots(len(fnArraList), 1, shared_xaxes='all',
                            shared_yaxes='all', horizontal_spacing=0.01,
                            vertical_spacing=0.01)
    else:
        raise ValueError(
            f"orientation must be 'v' or 'h' but is {orientation}")

    for cc, fnArr in enumerate(fnArraList):
        showlegend = True if cc == 0 else False
        traces = plot_ctx_clusters(fnArr, idxr=idxr, trial_mode=trial_mode,
                                   jitter=jitter, n_std=n_std,
                                   showlegend=showlegend)['data']
        if orientation == 'h':
            rows = [1] * len(traces)
            cols = [cc + 1] * len(traces)
        elif orientation == 'v':
            rows = [cc + 1] * len(traces)
            cols = [1] * len(traces)

        _ = fig.add_traces(traces, rows=rows, cols=cols)

    # zero lines
    fig.update_layout(template="simple_white", )
    fig.add_vline(0, line_width=2, line_color='black', line_dash='dash',
                  opacity=0.5)
    fig.add_hline(0, line_width=2, line_color='black', line_dash='dash',
                  opacity=0.5)

    # axis labels
    if orientation == 'h':
        xrow, xcol = None, None
        yrow, ycol = 1, 1
    elif orientation == 'v':
        xrow, xcol = len(fnArraList), 1
        yrow, ycol = None, None

    _ = fig.update_xaxes(title=dict(text='neuron 1 activity (AU)', standoff=0),
                         row=xrow, col=xcol)
    _ = fig.update_yaxes(scaleanchor='x', scaleratio=1,
                         title=dict(text='neuron 2 activity (AU)', standoff=0),
                         row=yrow, col=ycol)

    return fig


def plot_simple_psths(fnArrs: list[np.array], cellids: list[str],
                      cells: list[str], contexts: list[int], probes: list[int],
                      part: str = 'probe', avg: str = 'all',
                      color_palette: list[str] = FOURCOLOR) -> go.Figure:
    cidxs = [cellids.index(c) for c in cells]
    prbidxs = [p - 1 for p in probes]

    PSTHs = [a.mean(axis=0) for a in fnArrs]
    chn, ctx, prb, tme = PSTHs[0].shape

    if part == 'probe':
        t = np.linspace(0, 1, tme, endpoint=False)
    elif part == 'all':
        t = np.linspace(-1, 1, tme, endpoint=False)
    else:
        raise ValueError("parameter part must be 'all' or 'probe'")

    dashings = ['dot', 'dash', 'solid', 'dashdot']

    # rows, cols = square_rows_cols(len(cellid))
    fig = make_subplots(len(cells), len(fnArrs), shared_xaxes='all',
                        shared_yaxes='all', vertical_spacing=0.01,
                        horizontal_spacing=0.01)

    pan = 0
    for col, psth in enumerate(PSTHs):
        for row, cidx in enumerate(cidxs):
            # row, col = int(np.floor(cc / cols)) + 1, (cc % cols) + 1
            showlegend = True if pan == 0 else False
            pan = + 1

            # individual context probe lines
            for prbidx in prbidxs:
                dash = dashings[prbidx]

                for ctxidx in contexts:
                    color = color_palette[ctxidx]
                    y = psth[cidx, ctxidx, prbidx, :]
                    # tt, yy = squarefy(t, y)
                    # tt, yy = t, y
                    tt = np.linspace(t.min(), t.max(), 100)
                    yy = make_interp_spline(t, y, k=3)(tt)

                    fig.add_trace(go.Scatter(x=tt, y=yy, mode='lines',
                        line=dict(color=color, dash=dash, width=2),
                        name=f"ctx{ctxidx}_prb{prbidx + 1}",
                        showlegend=showlegend), row=row + 1, col=col + 1)

            # probe PSTH, all contexts averaged
            # in new for loop to ensure averages are on top of instace lines
            for prbidx in prbidxs:
                dash = dashings[prbidx]

                if avg == 'all':
                    y = psth[cidx, :, prbidx, :].mean(axis=0)
                elif avg == 'visible':
                    y = psth[cidx, contexts, prbidx, :].mean(axis=0)
                else:
                    raise ValueError(
                        "parameter avg must be 'all' or 'visible'")
                # tt, yy = squarefy(t, y)
                # tt, yy = t, y
                tt = np.linspace(t.min(), t.max(), 100)
                yy = make_interp_spline(t, y, k=3)(tt)
                fig.add_trace(go.Scatter(x=tt, y=yy, mode='lines',
                                         line=dict(color='black', dash=dash,
                                                   width=2),
                                         name=f"prb{prbidx + 1}_avg",
                                         showlegend=showlegend), row=row + 1,
                              col=col + 1)

            # add neuron name on left column
            if col == 0:
                # skips last row since its treated differently
                if row < len(cells) - 1:
                    _ = fig.update_yaxes(title_text=f'{cells[row]}<br> ',
                                         title_standoff=0, row=row + 1, col=1)

    _ = fig.update_xaxes(title_text='time from probe onset (s)',
                         title_standoff=0, row=len(cells), col=1)
    _ = fig.update_yaxes(title_text=f'{cells[-1]}<br>firing rate (z-score)',
                         title_standoff=0, row=len(cells), col=1)

    fig.update_layout(template='simple_white',
                      margin=dict(b=10, l=10, r=10, t=10))

    return fig
