import itertools as itt
import re

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import sem
from sklearn.decomposition import PCA

import src.models.modelnames as mns
from nems.db import batch_comp
from src.data.load import get_batch_ids
from src.data.rasters import load_site_formated_raster
from src.dim_redux.PCA import load_site_formated_PCs
from src.metrics.consolidated_tstat import tstat_cluster_mass
from src.metrics.delta_fr import pairwise_delta_FR
from src.metrics.significance import _significance
from src.models.param_tools import get_population_weights, get_strf, get_population_influence, get_pred_err, \
    model_independence_comparison, load_cell_formated_resp_pred
from src.visualization.fancy_plots import squarefy
from src.visualization.palette import *


def plot_raw_pair(cellid, contexts, probe, mode='psth', raster_fs=30, colors=FOURCOLOR, errortype='std',
                  pupil=False, simplify=False, part='all', mod_disp='both', fill_between=False, error_opacity=0.2,
                  **kwargs):
    prb_idx = probe - 1  # probe names start at 1 but we have zero idex

    if 'modelname' in kwargs.keys() and 'batch' in kwargs.keys():
        is_pred = True
        fs = int(re.findall('\.fs\d*\.', kwargs['modelname'])[0][3:-1])
        if raster_fs != fs: print("enforcing model raster_fs")

        site_raster, pred_raster, goodcells = load_cell_formated_resp_pred(cellid, part='all',
                                                                           **kwargs)
        # force PSTH
        if mode != 'psth':
            print('can only plot psth for predictions, forcing...')
            mode = 'psth'
        if pupil is not False:
            raise NotImplementedError('cannot make pupil distinction for model predicitonse')

    else:
        is_pred = False
        if mode == 'psth':
            fs = raster_fs
            smoothing_window = 50
        elif mode == 'raster':
            if raster_fs < 100:
                print(f'raster_fs={raster_fs} is too low for a good scatter. defaulting to 100hz')
                fs = 100
            else:
                fs = raster_fs
            smoothing_window = 0
        else:
            raise ValueError("undefined plot type, choose psth or raster")

        if "-PC-" in cellid:
            site_raster, goodcells = load_site_formated_PCs(cellid[:7], part='all',
                                                            smoothing_window=smoothing_window, raster_fs=fs)
            goodcells = list(goodcells.keys())
        else:
            site_raster, goodcells = load_site_formated_raster(cellid[:7], part='all',
                                                               smoothing_window=smoothing_window, raster_fs=fs)

        if pupil:
            pup_raster, _ = load_site_formated_raster(cellid[:7], part='all',
                                                      smoothing_window=0, raster_fs=fs,
                                                      pupil=True)

            pup_raster = np.mean(pup_raster, axis=-1, keepdims=True)
            pup_thresh = np.median(pup_raster, axis=0, keepdims=True)

            if pupil == 'big':
                pupil_mask = np.broadcast_to(pup_raster < pup_thresh, site_raster.shape)
            elif pupil == 'small':
                pupil_mask = np.broadcast_to(pup_raster >= pup_thresh, site_raster.shape)
            else:
                raise ValueError(f"pupil parameter must be False, 'big' or 'small. receivede {pupil}")

            site_raster = np.ma.masked_where(pupil_mask, site_raster, copy=False)

    eg_raster = site_raster[:, goodcells.index(cellid), :, prb_idx, :]
    if is_pred:
        eg_pred = pred_raster[:, goodcells.index(cellid), :, prb_idx, :]

    nreps, _, nsamps = eg_raster.shape
    duration = nsamps / fs
    time = np.linspace(0 - duration / 2, duration / 2, nsamps, endpoint=False)

    if part == 'all':
        halfs = [np.s_[:int(nsamps / 2)], np.s_[int(nsamps / 2):]]
    elif part == 'probe':
        halfs = [np.s_[int(nsamps / 2):]]
    else:
        raise ValueError(f'undefined value for part paramete: {part}')

    fig = go.Figure()
    for cc, ctx_idx in enumerate(contexts):

        if is_pred:
            part_color = [colors[ctx_idx % len(colors)], colors[ctx_idx % len(colors)]]
        else:
            if simplify is False:
                # probe and context lines have different colors since areas color help identify probes
                part_color = [colors[ctx_idx % len(colors)], colors[probe % len(colors)]]
            elif simplify is True:
                # the color asociated with the context is also used for the probe
                part_color = [colors[ctx_idx % len(colors)], colors[ctx_idx % len(colors)]]
            else:
                raise ValueError(f'simplify must be bool but is {simplify}')

        for nn, (half, color) in enumerate(zip(halfs, part_color)):

            if mode == 'psth':
                # find mean and standard error of the mean for line and confidence interval
                mean_resp = np.mean(eg_raster[:, ctx_idx, :], axis=0)
                if errortype == 'std':
                    err_resp = np.std(eg_raster[:, ctx_idx, :], axis=0)
                elif errortype == 'sem':
                    err_resp = sem(eg_raster[:, ctx_idx, :], axis=0)
                else:
                    raise ValueError(f"Unknown errortype value {errortype}. Use 'std' or 'sem'")

                # Labels the line with
                if nn == 0 and part == 'all':
                    name = f'context {ctx_idx}'
                else:
                    name = f'probe {probe} after context {ctx_idx}'

                x, y = squarefy(time[half], mean_resp[half])
                _, ystd = squarefy(time[half], err_resp[half])

                # confidence interval for real data, and prediciton for model fits
                if is_pred and (mod_disp in ['pred', 'both']):
                    mean_pred = np.mean(eg_pred[:, ctx_idx, :], axis=0)

                    name = f'{name} prediction'
                    xp, yp = squarefy(time[half], mean_pred[half])

                    if part == 'all' and nn == 1:
                        xp = np.insert(xp, 0, xp[0])
                        yp = np.insert(yp, 0, mean_resp[halfs[0]][-1])

                    # fill the area between context effects to highlight difference!
                    if fill_between and part == 'probe' and cc == 1:
                        rgb = hex_to_rgb(Grey)  # tuple
                        fill_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {error_opacity})'
                        _ = fig.add_trace(go.Scatter(x=xp, y=yp, mode='lines',
                                                     line=dict(color='rgba(0,0,0,0)'),
                                                     fill='tonexty', fillcolor=fill_color,
                                                     name='difference', showlegend=False))

                    _ = fig.add_trace(go.Scatter(x=xp, y=yp, mode='lines',
                                                 line=dict(color=color, width=3, dash='dot'),
                                                 name=name, showlegend=True))

                elif not is_pred and not fill_between:
                    # shadow of confidence interval for data with multiple trials
                    rgb = hex_to_rgb(part_color[0])  # tuple
                    fill_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {error_opacity})'
                    line_color = 'rgba(0,0,0,0)'  # transparent line in case its width is changed later outside this func

                    _ = fig.add_trace(go.Scatter(x=x, y=y + ystd, mode='lines', line_color=line_color, line_width=0,
                                                 showlegend=False))
                    _ = fig.add_trace(go.Scatter(x=x, y=y - ystd, mode='lines', line_color=line_color, line_width=0,
                                                 fill='tonexty', fillcolor=fill_color, showlegend=False))

                ## Main PSHT line ##
                # set the mean lines second so they lie on top of the colored areas

                if not (is_pred and mod_disp == 'pred'):

                    # for the second half prepend the last sample of the first half to create a connector
                    if part == 'all' and nn == 1:
                        x = np.insert(x, 0, x[0])
                        y = np.insert(y, 0, mean_resp[halfs[0]][-1])

                    if fill_between and part == 'probe' and cc == 1:
                        rgb = hex_to_rgb(Grey)  # tuple
                        fill_color = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {error_opacity})'
                        _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                                     # line=dict(color='rgba(0,0,0,0)'),
                                                     line=dict(color='black'),
                                                     fill='tonexty', fillcolor=fill_color,
                                                     name='difference', showlegend=False))

                    _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color, line_width=3,
                                                 name=name, showlegend=True))



            elif mode == 'raster':
                y, x = np.where(eg_raster[:, ctx_idx, half] > 0)
                x_offset = time[half][0]
                x = (x / fs) + x_offset
                y_offset = nreps * cc
                y += y_offset

                _ = fig.add_trace(
                    go.Scatter(x=x, y=y, mode='markers',
                               marker=dict(
                                   color=color,
                                   opacity=0.5,
                                   size=5,
                                   line=dict(
                                       color=part_color[0],
                                       width=0.5
                                   )
                               ),
                               showlegend=False
                               )
                )
            else:
                raise ValueError("undefined plot type, choose psht or raster")

    if part == 'all':
        x_range = [0 - duration / 2, duration / 2]
    elif part == 'probe':
        x_range = [0, duration / 2]
    else:
        raise ValueError(f'undefined value for part paramete: {part}')

    _ = fig.update_xaxes(title_text='time from probe onset (s)', title_standoff=0,
                         range=x_range)
    _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1)

    if mode == 'psth':
        _ = fig.update_yaxes(title_text='firing rate (z-score)', title_standoff=0)
    elif mode == 'raster':
        _ = fig.update_yaxes(title_text='trials', title_standoff=0, showticklabels=False, range=[0, nreps * 2])
    fig.update_layout(template='simple_white')

    return fig


def plot_pupil_so_effects(cellid, contexts, probe, raster_fs=30, error_opacity=0.2, ):
    prb_idx = probe - 1
    fs = raster_fs
    smoothing_window = 50

    site_raster, goodcells = load_site_formated_raster(cellid[:7], part='probe',
                                                       smoothing_window=smoothing_window, raster_fs=fs)

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

    for pp, (pup_size, dash, pup_mask) in enumerate(zip(['big', 'small'],
                                                        ['solid', 'dot'],
                                                        [big_mask, small_mask])):

        pup_raster = np.ma.masked_where(pup_mask, sel_raster, copy=False)
        PSTHs = np.mean(pup_raster, axis=0)

        # delta FR
        DFR = PSTHs[0, :] - PSTHs[1, :]

        # confidence interval: standard error of the difference
        std = np.sqrt(
            (sem(np.ma.compress_nd(pup_raster[:, 0, :], axis=0), axis=0) ** 2) +
            (sem(np.ma.compress_nd(pup_raster[:, 1, :], axis=0), axis=0) ** 2)
        )

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
        line_color = 'rgba(0,0,0,0)'  # transparent line in case its width is changed later outside this func

        _ = fig.add_trace(go.Scatter(x=x, y=y + yerr, mode='lines', line_color=line_color, line_width=0,
                                     showlegend=False))
        _ = fig.add_trace(go.Scatter(x=x, y=y - yerr, mode='lines', line_color=line_color, line_width=0,
                                     fill='tonexty', fillcolor=fill_color, showlegend=False))

        # meand Delta firing rate
        _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                     line=dict(color='black',
                                               dash=dash,
                                               width=3),
                                     name=pup_size, showlegend=True))

    _ = fig.update_xaxes(title_text='time from probe onset (s)', title_standoff=0)
    _ = fig.update_yaxes(title_text='delta firing rate (z-score)', title_standoff=0)
    fig.update_layout(template='simple_white')

    return fig


def plot_pop_modulation(cellid, modelname, batch, contexts, probe, **kwargs):
    fs = int(re.findall('\.fs\d*\.', modelname)[0][3:-1])
    _, mod_raster = get_population_influence(cellid, batch, modelname, **kwargs)

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
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color, line_width=3,
                                 line_dash='dash',
                                 name=name, showlegend=False))

    _ = fig.update_xaxes(title_text='time from probe onset (s)', title_standoff=0,
                         range=[0 - duration / 2, duration / 2])
    _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1)
    _ = fig.update_yaxes(title_text='pop modulation', title_standoff=0)

    return fig


def plot_pop_stategain(cellid, modelname, batch, orientation='v'):
    mean_pop_gain = get_population_weights(cellid, batch, modelname)
    if orientation == 'v':
        toplot = mean_pop_gain[:, None]
    elif orientation == 'h':
        toplot = mean_pop_gain[None, :]
    else:
        raise ValueError(f"unknown orientation value: {orientation}\nchoose 'v' or 'h'")

    img = go.Figure()
    # _ = img.add_trace(go.Heatmap(z=toplot, colorscale='BrBG', zmid=0))
    _ = img.add_trace(go.Heatmap(z=toplot, coloraxis="coloraxis"))
    # img = px.imshow(toplot, aspect='auto', color_continuous_scale='inferno')
    return img


def plot_mod_full(cellid, modelname, batch, contexts, probe, orientation='h', **kwargs):
    mod_plot = plot_pop_modulation(cellid, modelname, batch, contexts, probe)

    if orientation == 'h':
        fig = make_subplots(1, 2, column_widths=[0.95, 0.05], horizontal_spacing=0.01)
        # modulation
        fig.add_traces(mod_plot['data'], rows=[1] * len(mod_plot['data']), cols=[1] * len(mod_plot['data']))

        # weigts
        weight_plot = plot_pop_stategain(cellid, modelname, batch, orientation='v')
        fig.add_traces(weight_plot['data'], rows=[1] * len(weight_plot['data']), cols=[2] * len(weight_plot['data']))

        fig.update_layout(coloraxis=dict(colorscale='inferno',
                                         colorbar=dict(
                                             orientation='v',
                                             thickness=10, len=0.6,
                                             title_text='weight',
                                             title_side='right',
                                             tickangle=0,
                                             xanchor='left')
                                         ))

    elif orientation == 'v':
        fig = make_subplots(2, 1, row_width=[0.05, 0.95], vertical_spacing=0.01)
        # modulation
        fig.add_traces(mod_plot['data'], rows=[1] * len(mod_plot['data']), cols=[1] * len(mod_plot['data']))

        # weigts
        weight_plot = plot_pop_stategain(cellid, modelname, batch, orientation='h')
        fig.add_traces(weight_plot['data'], rows=[2] * len(weight_plot['data']), cols=[1] * len(weight_plot['data']))

        fig.update_layout(coloraxis=dict(colorscale='inferno',
                                         colorbar=dict(
                                             orientation='v',
                                             thickness=10, len=0.6,
                                             title_text='weight',
                                             title_side='right',
                                             tickangle=0,
                                             xanchor='left')
                                         ))

    _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1,
                      row=1, col=1)
    fig.update_layout(xaxis2=dict(tickvals=[], ticktext=[]),
                      yaxis2=dict(tickvals=[], ticktext=[])
                      )

    return fig


def plot_strf(cellid, modelname, batch):
    strf = get_strf(cellid, batch, modelname)
    img = px.imshow(strf, origin='lower', aspect='auto', color_continuous_scale='inferno')

    return img


def plot_errors_over_time(cellid, modelname, batch, contexts, probe, part='probe', grand_mean=False):
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
        # take neuron mean accuracy for the cell. dont blame me, Stephen made me do it!

        # first plot individual contex-probe error
        toplot = err.mean(axis=(0, 1))
        x, y = squarefy(time, toplot)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='gray', line_width=2, line_dash='dash',
                                 name='ctx_prb_mean', showlegend=True))

        # second add the difference error
        toplot = diff_err.mean(axis=(0, 1))
        x, y = squarefy(time, toplot)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='black', line_width=3,
                                 name='ctx-pair_prb_mean', showlegend=True))
    elif grand_mean is False:
        colors = [Grey, Yellow, Red, Teal, Brown]
        # first plot individual contex-probe error
        for cc, ctx_idx in enumerate(contexts):
            color = colors[ctx_idx % len(colors)]
            name = f'cxt{ctx_idx}_prb{probe}'
            toplot = err[ctx_idx, pidx, :]

            x, y = squarefy(time, toplot)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color, line_width=2, line_dash='dash',
                                     name=name, showlegend=True))

        # second add the difference error
        name = f'cxts{contexts}_prb{probe}'
        pair_idx = ctx_pairs.index(contexts)
        toplot = diff_err[pair_idx, pidx, :]

        x, y = squarefy(time, toplot)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='black', line_width=3,
                                 name=name, showlegend=True))
    else:
        raise ValueError(f"'grand_mean' should be bool but is {type(grand_mean)}")

    fig.update_layout(xaxis_title_text='time (ms)', yaxis_title_text='sqr error')

    return fig


def plot_multiple_errors_over_time(cellid, modelnames, batch, contexts, probe, part='probe', style='mean', floor=None,
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
            time = np.linspace(0 - duration / 2, duration / 2, tme, endpoint=False)

        color = colors[mm % len(colors)]
        if style == 'mean':
            name = f'{modelnicknames[modelname]}'
            toplot = diff_err.mean(axis=(0, 1))
            x, y = squarefy(time, toplot)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color, line_width=3,
                                     name=name, showlegend=True))
        elif style == 'instance':
            name = f'{modelnicknames[modelname]}_C{contexts}_P{probe}'
            pair_idx = ctx_pairs.index(tuple(contexts))
            toplot = diff_err[pair_idx, pidx, :]

            x, y = squarefy(time, toplot)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color, line_width=3,
                                     name=name, showlegend=True))

        elif style == 'PCA':

            pca = PCA(n_components=50)
            tofit = diff_err.reshape((-1, diff_err.shape[-1]))
            toplot = pca.fit_transform(tofit.T).T  # PC by Time

            dashings = ['solid', 'dash', 'dot']
            for pc in range(nPCs):
                name = f'{modelnicknames[modelname]}_PC{pc}'
                x, y = squarefy(time, toplot[pc, :])
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                         line=dict(color=color,
                                                   width=3,
                                                   dash=dashings[pc % len(dashings)]),
                                         name=name, showlegend=True))

                # inset for variance explained
                fig.add_trace(go.Scatter(x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                                         y=np.cumsum(pca.explained_variance_ratio_),
                                         mode='lines+markers', line_color=color, marker_color=color,
                                         xaxis='x2', yaxis='y2', showlegend=False)
                              )

            fig.update_layout(xaxis2=dict(domain=[0.7, 0.95], anchor='y2', title_text='PC#'),
                              yaxis2=dict(domain=[0.7, 0.95], anchor='x2', title_text='var explained')
                              )

        else:
            raise ValueError(f"{style} unrecognized 'style' value. Use 'instance', 'mean' or 'PCA'")

        fig.update_layout(xaxis_title_text='time (ms)', yaxis_title_text='sqr error')

    return fig


def plot_model_prediction_comparison(cellid, batch, independent_models, dependent_model, contexts, probe,
                                     part='probe', grand_mean=False):
    modelnicknames = {val: key for key, val in mns.modelnames.items()}
    colors = [Grey, Yellow, Red, Teal, Brown]

    rasters, aggs = model_independence_comparison(cellid, batch, independent_models, dependent_model, part=part)

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
                   rasters[independent_models[1]][ctx_idx, pidx, :],
                   (rasters[independent_models[0]] + rasters[independent_models[1]])[ctx_idx, pidx, :],
                   rasters[dependent_model][ctx_idx, pidx, :]]
        names = [f"{modelnicknames[independent_models[0]]}",
                 f"{modelnicknames[independent_models[1]]}",
                 'model_sum',
                 f"{modelnicknames[dependent_model]}"]
        dashings = ['dash',
                    'dot',
                    'dashdot',
                    'solid']
        widths = [1, 1, 3, 3.5]

        color = colors[ctx_idx % len(colors)]
        for toplot, name, dashing, width in zip(toplots, names, dashings, widths):
            x, y = squarefy(time, toplot)
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color, line_dash=dashing, line_width=width,
                                     name=name, showlegend=True),
                          row=1, col=cc + 1)

    # all context probes for the neuron
    to_concat = list()
    for pred_source in ['dependent', 'sum']:
        df = pd.DataFrame(index=range(ctx), columns=range(1, prb + 1), data=aggs[pred_source])
        df['model'] = pred_source
        to_concat.append(df)

    df = pd.concat(to_concat, axis=0)
    df.index.name = 'context'
    df.columns.name = 'probe'
    df.reset_index(inplace=True)

    toplot = df.melt(id_vars=['context', 'model']
                     ).pivot_table(index=['context', 'probe'], columns='model', values='value', aggfunc='first'
                                   ).reset_index()
    scatter = px.scatter(toplot, x='dependent', y='sum', hover_data=['context', 'probe'],
                         color_discrete_sequence=['black'])

    fig.add_traces(scatter['data'], rows=[1] * len(scatter['data']), cols=[3] * len(scatter['data']))

    # highlighs the selected  values
    selected = toplot.query(f"context in {contexts} and probe == {probe}")
    fig.add_trace(go.Scatter(x=selected['dependent'], y=selected['sum'], mode='markers',
                             marker_color=[colors[int(cc) % len(colors)] for cc in selected['context']],
                             marker_size=10),
                  row=1, col=3)

    # unit line
    plot_range = [toplot.loc[:, ['dependent', 'sum']].values.min(), toplot.loc[:, ['dependent', 'sum']].values.max()]
    fig.add_trace(go.Scatter(x=plot_range, y=plot_range, mode='lines', line_color='black', line_dash='dash'),
                  row=1, col=3)

    fig.update_xaxes(title_text=modelnicknames[dependent_model],
                     col=3, row=1)
    fig.update_yaxes(scaleratio=1,
                     title_text=f'{modelnicknames[independent_models[0]]} + {modelnicknames[dependent_model][0]}',
                     col=3, row=1)

    return fig


def plot_time_ser_quant(cellid, contexts, probe,
                        multiple_comparisons_axis, cluster_threshold,
                        alpha=0.05, source='real', secondary_y=False,
                        deltaFR=False, ignore_quant=False, meta={}):
    """
    plot shoing the quantification of time series differences (PSTHs) between context effects.
    it shows the difference metric (t-score), its threshold for cluster deffinition, the t-score sume for each cluster,
    and the threshold for cluster significance based on the spermutation distribution.
    It also displays the are of time bins in clusters that are significant, alongside the center of mass of this
    significant area.
    """
    raster_meta = {'montecarlo': 11000,
                   'raster_fs': 30,
                   'reliability': 0.1,
                   'smoothing_window': 0,
                   'stim_type': 'permutations',
                   'zscore': True}

    raster_meta.update(meta)
    montecarlo = raster_meta.pop('montecarlo')

    if "-PC-" in cellid:
        load_fn = 'PCA'
    else:
        load_fn = 'SC'

    if tstat_cluster_mass.check_call_in_cache(
            cellid[:7], cluster_threshold=float(cluster_threshold), montecarlo=montecarlo, raster_meta=raster_meta,
            load_fn=load_fn):
        dprime, pval_quantiles, goodcells, shuffled_eg = tstat_cluster_mass(
            cellid[:7], cluster_threshold=float(cluster_threshold), montecarlo=montecarlo, raster_meta=raster_meta,
            load_fn=load_fn)
    else:
        raise ValueError(f'{cellid[:7]}, {tstat_cluster_mass}, {cluster_threshold} not yet in cache')

    # uses the absolute delta FR to calculate integral
    if deltaFR:
        dfr = pairwise_delta_FR(cellid[:7], raster_meta=raster_meta, load_fn=load_fn)

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
        # asumes correction acros context and probes
        mult_comp = 'bf_cp'
    elif len(multiple_comparisons_axis) == 3:
        # asumes correction acros neurons, context and probes
        mult_comp = 'bf_ncp'
    else:
        raise ValueError('I dont know what to do with so many multiple_comparisons_axis')

    if type(goodcells) is dict:
        goodcells = list(goodcells.keys())
    cell_idx = goodcells.index(cellid)
    pair_idx = [f'{t0}_{t1}' for t0, t1 in itt.combinations(range(dprime.shape[2] + 1), 2)].index(
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
        print(f'using t-score threshold for sample-alpha {cluster_threshold} -> t = {CTT}')
    else:
        CTT = cluster_threshold

    SIG = significance[cell_idx, pair_idx, prb_idx, :]

    signif_mask = SIG > 0
    t = np.linspace(0, DP.shape[-1] / raster_meta['raster_fs'], DP.shape[-1], endpoint=False)

    # calculates integral, center of mass and last bin
    if not ignore_quant:
        if deltaFR:
            to_quantify = dfr
        else:
            to_quantify = DP

        integral = np.sum(np.abs(to_quantify[signif_mask])) * np.mean(np.diff(t))
        print(f"integral: {integral * 1000:.2f} t-score*ms")

        mass_center = np.sum(np.abs(to_quantify[signif_mask]) * t[signif_mask]) / np.sum(
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

    main_traces.append(go.Scatter(x=tt, y=mmdd, mode='lines', line_color='black', line_width=3,
                                  name=linename))
    main_traces.append(go.Scatter(x=tt[[0, -1]], y=[CTT] * 2, mode='lines',
                                  line=dict(color='Black', dash='dash', width=2),
                                  name='T-stat thresold'))

    if deltaFR:
        # here adds the T-score values as a dotted line
        tt, ttss = squarefy(t, DP)
        main_traces.append(go.Scatter(x=tt, y=ttss, mode='lines',
                                      line=dict(color='black',
                                                dash='dot',
                                                width=3),
                                      name='T-statistic'))

    ## significant area under the curve
    if not ignore_quant:
        _, smm = squarefy(t, signif_mask)
        wmmdd = np.where(smm, mmdd,
                         0)  # little hack to add gaps into the area, set y value to zero where no significance
        rgb = hex_to_rgb(AMPCOLOR)
        rgba = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.5)'

        main_traces.append(go.Scatter(x=tt, y=wmmdd, mode='none',
                                      fill='tozeroy', fillcolor=rgba,
                                      name='integral'))

        ## center of mass indication: line fom zero to the time series value at that time point
        if not np.isnan(mass_center):
            ytop = DP[np.abs(t - mass_center).argmin()]
            main_traces.append(go.Scatter(x=[mass_center] * 2, y=[0, ytop], mode='lines',
                                          line=dict(color=DURCOLOR, width=4),
                                          name='center of mass'))

        ## adds star at the last time bin position
        if last_bin > 0:
            main_traces.append(go.Scatter(x=[last_bin], y=[0], mode='markers',
                                          marker=dict(symbol='star',
                                                      color=DURCOLOR,
                                                      size=15),
                                          name='last bin'))

    for trace in main_traces:
        fig.add_trace(trace, secondary_y=False)  # forces main traces to be on the primary y ax

    # cluster and corrected confidence interval of the shuffled clusters, this can be on a secondary y axis
    tt, mmcc = squarefy(t, CT)
    secondary_traces.append(go.Scatter(x=tt, y=mmcc, mode='lines',
                                       line=dict(color=Green, dash='solid', width=3),
                                       name='cluster sum'))
    secondary_traces.append(go.Scatter(x=tt[[0, -1]], y=[CI] * 2, mode='lines',
                                       line=dict(color=Green, dash='dash', width=2),
                                       name='cluster threshold'))

    for trace in secondary_traces:
        fig.add_trace(trace, secondary_y=secondary_y)

    # formats axis, legend and so on.
    _ = fig.update_xaxes(title=dict(text='time from probe onset (s)', standoff=0))

    _ = fig.update_yaxes(title=dict(text="difference (t-score)", standoff=0))
    if secondary_y:
        _ = fig.update_yaxes(title=dict(text="cluster sum (t-score)", standoff=0), secondary_y=secondary_y)

    return fig, main_traces, secondary_traces


def plot_simple_quant(cellid, contexts, probe,
                      multiple_comparisons_axis, cluster_threshold,
                      alpha=0.05, meta={}):
    """
    plot shoing the quantification of time series differences (PSTHs) between context effects.
    it shows the difference metric (t-score), its threshold for cluster deffinition, the t-score sume for each cluster,
    and the threshold for cluster significance based on the spermutation distribution.
    It also displays the are of time bins in clusters that are significant, alongside the center of mass of this
    significant area.
    """
    raster_meta = {'montecarlo': 11000,
                   'raster_fs': 30,
                   'reliability': 0.1,
                   'smoothing_window': 0,
                   'stim_type': 'permutations',
                   'zscore': True}

    raster_meta.update(meta)
    montecarlo = raster_meta.pop('montecarlo')

    if "-PC-" in cellid:
        load_fn = 'PCA'
    else:
        load_fn = 'SC'

    if tstat_cluster_mass.check_call_in_cache(
            cellid[:7], cluster_threshold=float(cluster_threshold), montecarlo=montecarlo, raster_meta=raster_meta,
            load_fn=load_fn):
        _, pval_quantiles, goodcells, _ = tstat_cluster_mass(
            cellid[:7], cluster_threshold=float(cluster_threshold), montecarlo=montecarlo, raster_meta=raster_meta,
            load_fn=load_fn)
    else:
        raise ValueError(f'{cellid[:7]}, {tstat_cluster_mass}, {cluster_threshold} not yet in cache')

    # uses the absolute delta FR to calculate integral
    DFR = pairwise_delta_FR(cellid[:7], raster_meta=raster_meta, load_fn=load_fn)

    significance = _significance(pval_quantiles['pvalue'],
                                 multiple_comparisons_axis=multiple_comparisons_axis,
                                 alpha=alpha)

    if type(goodcells) is dict:
        goodcells = list(goodcells.keys())
    cell_idx = goodcells.index(cellid)
    pair_idx = [f'{t0}_{t1}' for t0, t1 in itt.combinations(range(DFR.shape[2] + 1), 2)].index(
        f'{contexts[0]}_{contexts[1]}')
    prb_idx = probe - 1

    # figures out if flip is neede
    DFR = DFR[cell_idx, pair_idx, prb_idx, :]
    DFR = np.abs(DFR)

    SIG = significance[cell_idx, pair_idx, prb_idx, :]

    signif_mask = SIG > 0
    t = np.linspace(0, DFR.shape[-1] / raster_meta['raster_fs'], DFR.shape[-1], endpoint=False)

    # calculates center of mass and integral
    integral = np.sum(np.abs(DFR[signif_mask])) * np.mean(np.diff(t))
    print(f"integral: {integral:.3f} Z-score*s")

    mass_center = np.sum(np.abs(DFR[signif_mask]) * t[signif_mask]) / np.sum(np.abs(DFR[signif_mask]))
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
                                  line=dict(color='black',
                                            width=3),
                                  name="delta FR"))

    ## significant area under the curve
    _, smm = squarefy(t, signif_mask)
    wmmdd = np.where(smm, mmdd, 0)  # little hack to add gaps into the area, set y value to zero where no significance
    rgb = hex_to_rgb(AMPCOLOR)
    rgba = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.5)'

    main_traces.append(go.Scatter(x=tt, y=wmmdd, mode='none',
                                  fill='tozeroy', fillcolor=rgba,
                                  name='integral'))

    ## center of mass indication: line fom zero to the time series value at that time point
    if not np.isnan(mass_center):
        ytop = DFR[np.abs(t - mass_center).argmin()]
        main_traces.append(go.Scatter(x=[mass_center] * 2, y=[0, ytop], mode='lines',
                                      line=dict(color=DURCOLOR, width=4),
                                      name='center of mass'))

    ## adds star at the last time bin position
    if last_bin > 0:
        main_traces.append(go.Scatter(x=[last_bin], y=[0], mode='markers',
                                      marker=dict(symbol='star',
                                                  color=DURCOLOR,
                                                  size=15),
                                      name='last bin'))

    fig.add_traces(main_traces)  # forces main traces to be on the primary y ax

    # formats axis, legend and so on.
    _ = fig.update_xaxes(title=dict(text='time from probe onset (s)', standoff=0))

    _ = fig.update_yaxes(title=dict(text="delta Firing Rate", standoff=0))

    return fig


def plot_tiling(picked_id, df, zmax=None, time_metric='last_bin',
                show_coloraxis=True, orientation='h', cscales=None):
    # turns long format data into an array with dimension Probe * context_pair
    if len(picked_id) == 7:
        # site case, get max projection across neurons
        to_pivot = df.query(f"site == '{picked_id}' and metric in ['integral', '{time_metric}']").groupby(
            ['metric', 'probe', 'context_pair']).agg(
            value=('value', 'max'))
    else:
        # neuron case, just select data
        to_pivot = df.query(f"id == '{picked_id}' and metric in ['integral', '{time_metric}']")
    val_df = to_pivot.pivot_table(index=['metric', 'probe'], columns=['context_pair'], values='value')

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

        colors = pc.sample_colorscale(cscales[metric],
                                      (val_df.loc[metric] / max_vals[metric]).values.flatten())
        color_df.loc[metric] = np.asarray(colors).reshape(color_df.loc[metric].shape)

    # general shapes of the upper and lower triangles to be passed to Scatter x and y
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
                            line_width=1, line_color='#222222',
                            fill='toself', fillcolor=amp_color[p, c],
                            marker=dict(color=(amplitudes[p, c],) * len(xu),
                                        coloraxis='coloraxis',
                                        opacity=0,
                                        cmin=0, cmax=max_vals['integral'],
                                        ),
                            showlegend=False
                            )

        # duration lower half
        _ = fig.add_scatter(x=xl + c, y=yl + p, mode='lines+markers',
                            line_width=1, line_color='#222222',
                            fill='toself', fillcolor=dur_color[p, c],
                            marker=dict(color=(durations[p, c],) * len(xl),
                                        coloraxis='coloraxis2',
                                        opacity=0,
                                        cmin=0, cmax=max_vals[time_metric],
                                        ),
                            showlegend=False
                            )

    # adds encasing margin
    h, w = amplitudes.shape
    x = [0, w, w, 0, 0]
    y = [0, 0, h, h, 0]
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                             line=dict(color='black',
                                       width=1),
                             showlegend=False)
                  )

    # label handling
    context_pairs = [f"{int(pp.split('_')[0])}_{int(pp.split('_')[1])}"
                     for pp in val_df.columns.to_list()]
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

    fig.update_xaxes(dict(scaleanchor='y',
                          constrain='domain',
                          range=xrange, fixedrange=True,
                          title_text=xtitle,
                          tickmode='array',
                          tickvals=xtickvals,
                          ticktext=xticktext,
                          showticklabels=True)
                     )
    fig.update_yaxes(dict(constrain='domain',
                          range=yrange, fixedrange=True,
                          title_text=f'{picked_id}<br>{ytitle}',
                          tickmode='array',
                          tickvals=ytickvals,
                          ticktext=yticktext,
                          showticklabels=True)

                     )
    # set the positions of the colorbars
    if show_coloraxis:
        fig.update_layout(coloraxis=dict(colorscale=cscales['integral'],
                                         colorbar=dict(
                                             thickness=10, len=0.6,
                                             title_text='integral',
                                             title_side='right',
                                             tickangle=-90,
                                             xanchor='left', x=1)
                                         ),
                          coloraxis2=dict(colorscale=cscales[time_metric],
                                          colorbar=dict(
                                              thickness=10, len=0.6,
                                              title_text=time_metric,
                                              title_side='right',
                                              tickangle=-90,
                                              xanchor='left', x=1.1)
                                          )
                          )
    else:
        fig.update_layout(
            coloraxis=dict(showscale=False),
            coloraxis2=dict(showscale=False)
        )

    fig.update_layout(template='simple_white')

    return fig


def plot_model_fitness(cellids, modelnames, nicknames=None, stat='r_test', mode='bars'):
    """
    parses data from CPN experiments (batchs) and models into a figure displaying r-test
    """
    DF = batch_comp(batch=[326, 327], modelnames=modelnames, cellids=cellids, stat=stat)
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

    wide = pd.merge(DF, regdf, on='cellid').rename(columns={'cellid': 'id', 'siteid': 'site'})

    if mode == 'bars':
        long = pd.melt(wide, id_vars=['id', 'site', 'region'], value_vars=mod_cols, var_name='model', value_name=stat)
        fig = px.box(long, x='model', y=stat, color='region', points='all')

    return fig


if __name__ == '__main__':
    from configparser import ConfigParser
    from src.root_path import config_path

    config = ConfigParser()
    config.read_file(open(config_path / 'settings.ini'))
    meta = {'reliability': 0.1,  # r value
            'smoothing_window': 0,  # ms
            'raster_fs': 30,
            'montecarlo': 11000,
            'zscore': True,
            'stim_type': 'permutations'}

    cellid, contexts, probe = 'TNC019a-042-5', (0, 3), 3
    cellid, contexts, probe = 'TNC019a-PC-1', (0, 3), 3
    cellid, contexts, probe = 'ARM021b-36-8', (0, 1), 3  # from paper figure examples
    cellid, contexts, probe = 'TNC014a-22-2', (0, 8), 3  # form paper modeling figure

    # # digested metric plots aka tile plots
    # df = jl.load(pl.Path(config['paths']['analysis_cache']) / f'220310_ctx_mod_metric_DF_tstat_cluster_mass_BS')
    # df = jl.load(pl.Path(config['paths']['analysis_cache']) / f'220520_minimal_DF')
    # df.query("source == 'real' and mult_comp_corr ==  'bf_cp'  and metric in ['integral', 'last_bin']", inplace=True)
    # tile = plot_tiling('ARM021b-36-8', df,
    #                    time_metric='last_bin',
    #                    show_coloraxis=False,
    #                    # cscales={'integral': 'Greens',
    #                    #          'last_bin': 'Purples'}
    #                    )
    # tile.show()
    #
    # tile = plot_tiling('ARM021b-36-8', df,
    #                    time_metric='last_bin',
    #                    show_coloraxis=False,
    #                    cscales={'integral': 'Greens',
    #                             'last_bin': 'Purples'}
    #                    )
    # tile.show()
    #
    # tile = plot_tiling('ARM021b-36-8', df,
    #                    time_metric='last_bin',
    #                    show_coloraxis=True,
    #                    orientation='h',
    #                    cscales={'integral': 'BuGn',
    #                             'last_bin': 'BuPu'})
    # tile.show()

    # # rawish data plots, aka psth, raster and quantification
    # fig = make_subplots(1,4)
    # raster = plot_raw_pair(cellid, contexts, probes, type='raster')
    # psth = plot_raw_pair(cellid, contexts, probe, type='psth', raster_fs=20, pupil='big')
    # psth.show()
    # psth = plot_raw_pair(cellid, contexts, probe, type='psth', raster_fs=20, pupil='small',simplify=True, error_opacity=0.1)
    # psth.show()

    # dfr = plot_pupil_so_effects(cellid, contexts, probe, raster_fs=30, error_opacity=0.2)
    # dfr.show()

    # quant0,_,_ = plot_time_ser_quant(cellid, contexts, probe, source='real',
    #                              multiple_comparisons_axis=[1, 2], cluster_threshold=0.05,secondary_y=True,
    #                              meta=dict(raster_fs=20))
    # quant0.show()

    # quant0,_,_ = plot_time_ser_quant(cellid, contexts, probe, source='real',
    #                              multiple_comparisons_axis=[1, 2], cluster_threshold=0.05,secondary_y=True,
    #                              meta=dict(raster_fs=20), deltaFR=False, ignore_quant=True)
    # quant0.show()
    #
    # quant0 = plot_simple_quant(cellid, contexts, probe,
    #                              multiple_comparisons_axis=[1, 2], cluster_threshold=0.05,
    #                              meta=dict(raster_fs=20))
    #
    # quant0.show()

    # quant1 = plot_time_ser_quant(cellid, contexts, probe, source='real',
    #                              multiple_comparisons_axis=[1,2], consecutive=0, cluster_threshold=0.05,
    #                              fn_name='big_shuff', meta={'montecarlo': 11000})
    # fig.add_traces(raster['data'],rows=[1]*len(raster['data']),cols=[1]*len(raster['data']))
    # fig.add_traces(psth['data'],rows=[1]*len(psth['data']),cols=[2]*len(psth['data']))
    # fig.add_traces(quant0['data'],rows=[1]*len(quant0['data']),cols=[3]*len(quant0['data']))
    # fig.add_traces(quant1['data'],rows=[1]*len(quant1['data']),cols=[4]*len(quant1['data']))
    # fig.show()

    # model parameter plots, aka pop_stategain etc.
    cellid = 'TNC014a-22-2'
    batch = 326
    modelname = "ozgf.fs100.ch18-ld.popstate-dline.15.15.1-norm-epcpn.seq-avgreps_" \
                "dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1-stategain.S.d_" \
                "jk.nf10-tfinit.n.lr1e3.et3.cont-newtf.n.lr1e4.cont-svpred"
    # fig = plot_pop_stategain(cellid, modelname, batch)
    # fig.show()
    # fig = plot_strf(cellid, modelname, batch)
    # fig.show()

    # raw prediction plot
    # psth_pred = plot_raw_pair(cellid, contexts, probe,
    #                           modelname=modelname, batch=batch,
    #                           part='probe', mod_disp='pred', fill_between=True)
    # psth_pred.show()

    # psth_pred = plot_raw_pair(cellid, contexts, probe, raster_fs=20)
    # psth_pred.show()

    # # population modulation
    # pop_mod = plot_pop_modulation(cellid, modelname, batch, contexts, probe)
    # pop_mod.show()

    # composite = go.Figure()
    # composite.add_traces(psth_pred.data)
    # composite.add_traces(pop_mod.data)
    # composite.show()

    # fig = plot_mod_full(cellid, modelname, batch, contexts, probe, orientation='v')
    # fig.show()

    # fig = plot_errors_over_time(cellid, modelname, batch, contexts, probe, grand_mean=False)
    # fig.show()

    # fig = plot_multiple_errors_over_time(cellid, [mns.STRF_relu, mns.pop_mod_relu, mns.self_mod_relu],batch, contexts, probe,
    #                                      part='probe', style='PCA', floor=mns.STRF_relu, nPCs=3)
    # fig.show()

    # fig  = plot_model_prediction_comparison(cellid, batch, [STRF_long_relu, pop_lone_relu], pop_mod_relu, contexts, probe,
    #                                  part='probe', grand_mean=False)
    # fig.show()

    # ###
    # from src.models.modelnames import modelnames
    # from src.utils.subsets import cellid_A1_fit_set, cellid_PEG_fit_set
    # mnames = {nick:modelnames[nick] for nick in ['matchl_STRF', 'matchl_self', 'matchl_pop','matchl_full']}
    # cellids = cellid_A1_fit_set.union(cellid_PEG_fit_set)
    # fig = plot_model_fitness(cellids, mnames.values(), nicknames=mnames.keys())
    # fig.show()
