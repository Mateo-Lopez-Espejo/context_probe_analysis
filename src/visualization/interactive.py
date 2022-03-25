import itertools as itt

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from webcolors import hex_to_rgb

from src.data.rasters import load_site_formated_raster, load_site_formated_prediction
from src.metrics.consolidated_dprimes import single_cell_dprimes_cluster_mass
from src.metrics.consolidated_mean_diff import single_cell_mean_diff_cluster_mass
from src.metrics.consolidated_tstat import single_cell_tstat_cluster_mass
from src.metrics.consolidated_tstat_big_shuff import single_cell_tstat_cluster_mass as big_shuff
from src.metrics.significance import _significance
from src.visualization.fancy_plots import squarefy
from src.visualization.palette import *

def plot_raw_pair(cellid, contexts, probe, type='psth', modelspec=None, **kwargs):
    probe -= 1 # probe names start at 1 but we have zero idex

    if modelspec is not None:
        fs = 100 # hard code for now for model fittings
        site_raster, goodcellse = load_site_formated_prediction(cellid[:7], part='all',
                                                                raster_fs=fs, modelspec=modelspec, cellid=cellid, **kwargs)
        # force PSTH
        if type != 'psth':
            print('can only plot psth for predictions, forcing...')
            type = 'psth'
    else:
        if type == 'psth':
            fs = 30 # dont pass as is default
            smoothing_window = 50
        elif type == 'raster':
            fs = 100
            smoothing_window = 0
        else:
            raise ValueError("undefined plot type, choose psht or raster")
        site_raster, goodcellse = load_site_formated_raster(cellid[:7], part='all',
                                                            smoothing_window=smoothing_window, raster_fs=fs)

    eg_raster = site_raster[:, goodcellse.index(cellid), :, probe, :]


    # fs = 30# here asuming 30Hz sampling rate, as its the default of the raster loader
    nreps,_,nsamps = eg_raster.shape
    duration  = nsamps / fs
    time = np.linspace(0-duration/2, duration/2, nsamps, endpoint=False)
    halfs = [np.s_[:int(nsamps / 2)], np.s_[int(nsamps / 2):]]

    # rotation of colors for the silence + 4 sound examples
    colors = [Grey, Yellow, Red, Teal, Brown]

    fig = go.Figure()
    for cc, ctx_idx in enumerate(contexts):

        part_color = [colors[ctx_idx % len(colors)], colors[probe % len(colors)]]

        for nn, (half, color) in enumerate(zip(halfs, part_color)):

            if type == 'psth':
                # find mean and estandard error of the mean for line and confidence interval
                mean_resp = np.mean(eg_raster[:, ctx_idx, :], axis=0)
                std_resp = np.std(eg_raster[:, ctx_idx, :], axis=0)
                x, y = squarefy(time[half], mean_resp[half])
                _, ystd = squarefy(time[half], std_resp[half])

                if nn == 0:
                    # same color of ci border line and fill for left-hand side
                    _ = fig.add_trace(go.Scatter(x=x, y=y + ystd, mode='lines', line_color=color, line_width=1))
                    _ = fig.add_trace(go.Scatter(x=x, y=y - ystd, mode='lines', line_color=color, line_width=1,
                                                 fill='tonexty'))

                else:
                    # different color of ci border line and fill for right-hand side
                    # to set a transparent fillcolor changes the 'rgb(x,y,z)' into 'rgba(x,y,z,a)'
                    rgb = hex_to_rgb(part_color[0])  # tuple
                    fill_opacity = 0.5
                    rgba = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {fill_opacity})'

                    _ = fig.add_trace(go.Scatter(x=x, y=y + ystd, mode='lines', line_color=color, line_width=1))
                    _ = fig.add_trace(go.Scatter(x=x, y=y - ystd, mode='lines', line_color=color, line_width=1,
                                                 fill='tonexty', fillcolor=rgba))

                # set the mean lines second so they lie on top of the colored areas
                _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color, line_width=3))

            elif type == 'raster':
                y, x = np.where(eg_raster[:, ctx_idx, half] > 0)
                x_offset = time[half][0]
                x = (x/fs) + x_offset
                y_offset = nreps * cc
                y += y_offset

                # set the mean lines second so they lie on top of the colored areas
                _ = fig.add_trace(
                    go.Scatter(x=x, y=y, mode='markers',
                                             marker=dict(
                                                 color=color,
                                                 opacity=0.5,
                                                 line=dict(
                                                     color=part_color[0],
                                                     width=1
                                                 )
                                             )
                               )
                )
            else:
                raise ValueError("undefined plot type, choose psht or raster")

    _ = fig.update_xaxes(title_text='time from probe onset (s)', title_standoff=0, range=[0-duration/2, duration/2])
    _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1)

    if type == 'psth':
        _ = fig.update_yaxes(title_text='firing rate (z-score)', title_standoff=0)
    elif type == 'raster':
        _ = fig.update_yaxes(title_text='trials', title_standoff=0, showticklabels=False, range=[0, nreps*2])
        fig.update_layout()

    return fig


def plot_predition(cellid, contexts, probe, modelspec=None, ctx=None):
    # todo work in progress, this should load models, calcupate predictions if neede and plot them
    ctx_pair = contexts
    prb_idx = probe - 1

    fs = 100

    site_raster, goodcellse = load_site_formated_prediction(ctx)

    eg_raster = site_raster[:, goodcellse.index(cellid), :, prb_idx, :]


    # fs = 30# here asuming 30Hz sampling rate, as its the default of the raster loader
    nreps,_,nsamps = eg_raster.shape
    duration  = nsamps / fs
    time = np.linspace(0-duration/2, duration/2, nsamps, endpoint=False)
    halfs = [np.s_[:int(nsamps / 2)], np.s_[int(nsamps / 2):]]

    # rotation of colors for the silence + 4 sound examples
    colors = [Grey, Yellow, Red, Teal, Brown]

    fig = go.Figure()
    for cc, ctx_idx in enumerate(ctx_pair):

        part_color = [colors[ctx_idx % len(colors)], colors[prb_idx % len(colors)]]

        for nn, (half, color) in enumerate(zip(halfs, part_color)):

            if type == 'psth':
                # find mean and estandard error of the mean for line and confidence interval
                mean_resp = np.mean(eg_raster[:, ctx_idx, :], axis=0)
                std_resp = np.std(eg_raster[:, ctx_idx, :], axis=0)
                x, y = squarefy(time[half], mean_resp[half])
                _, ystd = squarefy(time[half], std_resp[half])

                if nn == 0:
                    # same color of ci border line and fill for left-hand side
                    _ = fig.add_trace(go.Scatter(x=x, y=y + ystd, mode='lines', line_color=color, line_width=1))
                    _ = fig.add_trace(go.Scatter(x=x, y=y - ystd, mode='lines', line_color=color, line_width=1,
                                                 fill='tonexty'))

                else:
                    # different color of ci border line and fill for right-hand side
                    # to set a transparent fillcolor changes the 'rgb(x,y,z)' into 'rgba(x,y,z,a)'
                    rgb = hex_to_rgb(part_color[0])  # tuple
                    fill_opacity = 0.5
                    rgba = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {fill_opacity})'

                    _ = fig.add_trace(go.Scatter(x=x, y=y + ystd, mode='lines', line_color=color, line_width=1))
                    _ = fig.add_trace(go.Scatter(x=x, y=y - ystd, mode='lines', line_color=color, line_width=1,
                                                 fill='tonexty', fillcolor=rgba))

                # set the mean lines second so they lie on top of the colored areas
                _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color, line_width=3))

            elif type == 'raster':
                y, x = np.where(eg_raster[:, ctx_idx, half] > 0)
                x_offset = time[half][0]
                x = (x/fs) + x_offset
                y_offset = nreps * cc
                y += y_offset

                # set the mean lines second so they lie on top of the colored areas
                _ = fig.add_trace(
                    go.Scatter(x=x, y=y, mode='markers',
                                             marker=dict(
                                                 color=color,
                                                 opacity=0.5,
                                                 line=dict(
                                                     color=part_color[0],
                                                     width=1
                                                 )
                                             )
                               )
                )
            else:
                raise ValueError("undefined plot type, choose psht or raster")

    _ = fig.update_xaxes(title_text='time from probe onset (s)', title_standoff=0, range=[0-duration/2, duration/2])
    _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1)

    if type == 'psth':
        _ = fig.update_yaxes(title_text='firing rate (z-score)', title_standoff=0)
    elif type == 'raster':
        _ = fig.update_yaxes(title_text='trials', title_standoff=0, showticklabels=False, range=[0, nreps*2])
        fig.update_layout()

    return fig


def plot_time_ser_quant(cellid, contexts, probe,
                         multiple_comparisons_axis, consecutive, cluster_threshold, fn_name,
                         alpha=0.05, source='real', meta={}):
    defaults_meta = {'montecarlo': 1000,
                     'raster_fs': 30,
                     'reliability': 0.1,
                     'smoothing_window': 0,
                     'stim_type': 'permutations',
                     'zscore': True}


    fn_dict = {'dprime':single_cell_dprimes_cluster_mass,
               'mean_difference':single_cell_mean_diff_cluster_mass,
               't_statistic':single_cell_tstat_cluster_mass,
               'big_shuff': big_shuff}

    fn = fn_dict[fn_name]
    defaults_meta.update(meta)

    if fn.check_call_in_cache(cellid[:7], contexts='all', probes='all', cluster_threshold=float(cluster_threshold), meta=defaults_meta):
        dprime, pval_quantiles, goodcells, shuffled_eg = fn(
            cellid[:7], contexts='all', probes='all', cluster_threshold=float(cluster_threshold), meta=defaults_meta
        )
    else:
        raise ValueError(f'{cellid[:7]}, {fn}, {cluster_threshold} not yet in cache')


    if source == 'real':
        pvalue = pval_quantiles['pvalue']
    elif source == 'shuffled_eg':
        dprime = shuffled_eg['dprime']
        pvalue = shuffled_eg['pvalue']



    significance = _significance(pvalue,
                                 multiple_comparisons_axis=multiple_comparisons_axis,
                                 consecutive=consecutive,
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

    mult_comp


    cell_idx = goodcells.index(cellid) if len(cellid) > 7 else 0
    pair_idx = [f'{t0}_{t1}' for t0, t1 in itt.combinations(range(dprime.shape[2] + 1), 2)].index(
        f'{contexts[0]}_{contexts[1]}')
    prb_idx = probe - 1

    # figurese out if need flip
    # eg_idx = np.s_[cell_idx, pair_idx, prb_idx, :]
    DP = dprime[cell_idx, pair_idx, prb_idx, :]
    if np.sum(DP) < 0:
        flip = -1
    else:
        flip = 1

    DP *= flip
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
    t = np.linspace(0, DP.shape[-1] / defaults_meta['raster_fs'], DP.shape[-1], endpoint=False)

    # calculates center of mass and integral
    integral = np.sum(np.abs(DP[signif_mask])) * np.mean(np.diff(t))
    print(f"integral: {integral * 1000:.2f} d'*ms")

    mass_center = np.sum(np.abs(DP[signif_mask]) * t[signif_mask]) / np.sum(np.abs(DP[signif_mask]))
    if np.isnan(mass_center): mass_center = 0
    print(f'center of mass: {mass_center * 1000:.2f} ms')

    if np.any(signif_mask):
        last_bin = np.max(t[signif_mask])
    else:
        last_bin = 0
    # if np.isnan(significant_abs_mass_center): significant_abs_mass_center = 0
    print(f'last bin: {last_bin * 1000:.2f} ms')

    fig = go.Figure()
    # plots dprime and cluster threshold on primary axis
    tt, mmdd = squarefy(t, DP)
    _ = fig.add_trace(go.Scatter(x=tt, y=mmdd, mode='lines', line_color='black', line_width=3))
    _ = fig.add_trace(go.Scatter(x=tt[[0,-1]], y=[CTT]*2, mode='lines',
                                 line=dict(color='Black', dash='dash', width=2)))

    # cluster and corrected confidence interval of the shuffled clusters
    tt, mmcc = squarefy(t, CT)
    _ = fig.add_trace(go.Scatter(x=tt, y=mmcc, mode='lines', line_color=Green, line_dash='dot', line_width=3))
    _ = fig.add_trace(go.Scatter(x=tt[[0,-1]], y=[CI]*2, mode='lines',
                                 line=dict(color=Green, dash='dash', width=2)))

    # significant area under the curve
    # little hack to add gaps into the area, set d' value to zero where no significance
    _, smm = squarefy(t, signif_mask)
    wmmdd = np.where(smm, mmdd, 0)
    rgb = hex_to_rgb(Green)
    rgba = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.5)'

    _ = fig.add_trace(go.Scatter(x=tt, y=wmmdd, mode='none',
                                 fill='tozeroy', fillcolor=rgba))

    # center of mass indication: line fom zero to the time series value at that time point
    if not np.isnan(mass_center):
        ytop = DP[np.abs(t-mass_center).argmin()]
        _ = fig.add_trace(go.Scatter(x=[mass_center]*2, y=[0, ytop], mode='lines',
                                     line=dict(color=Purple, width=4)))

    # formats axis, legend and so on.
    _ = fig.update_xaxes(title=dict(text='time from probe onset (s)', standoff=0))

    _ = fig.update_yaxes(title=dict(text="contexts d'", standoff=0))

    return fig

def plot_tiling(picked_id, df):
    # turns long format data into an array with dimension Probe * context_pair

    if len(picked_id) == 7:
        #site case, get max projection across neurons
        to_pivot = df.query(f"site == '{picked_id}'").groupby(
            ['metric', 'probe', 'context_pair']).agg(
            value=('value','max'))
    else:
        #neuron case, just select data
        to_pivot = df.query(f"id == '{picked_id}'")
    val_df = to_pivot.pivot_table(index=['metric', 'probe'], columns=['context_pair'], values='value')

    cscales = {'integral': pc.make_colorscale(['#000000', Green]),
               'last_bin': pc.make_colorscale(['#000000', Purple])}
    max_vals = dict()
    # normalizes,saves max values and get colors for each metric
    color_df = val_df.copy()
    for metric in color_df.index.levels[0]:
        max_vals[metric] = val_df.loc[metric].values.max()
        colors = pc.sample_colorscale(cscales[metric],
                                      (val_df.loc[metric] / max_vals[metric]).values.flatten())
        color_df.loc[metric] = np.asarray(colors).reshape(color_df.loc[metric].shape)

    # general shapes of the upper and lower triangles to be passed to Scatter x and y
    xu, yu = np.array([0, 0, 1, 0]), np.array([0, 1, 1, 0])
    xl, yl = np.array([0, 1, 1, 0]), np.array([0, 0, 1, 0])

    amp_color = color_df.loc[('integral'), :].values
    dur_color = color_df.loc[('last_bin'), :].values

    amplitudes = val_df.loc[('integral'), :]
    durations = val_df.loc[('last_bin'), :]

    fig = go.Figure()

    for nn, (p, c) in enumerate(np.ndindex(amp_color.shape)):
        # note the use of transparent markers to define the colorbars internally
        # amplitud uppe half
        _ = fig.add_scatter(x=xu + c, y=yu + p, mode='lines+markers',
                            line_width=1, line_color='#222222',
                            fill='toself', fillcolor=amp_color[p, c],
                            marker=dict(color=(amplitudes.values[p, c],) * len(xu),
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
                            marker=dict(color=(durations.values[p, c],) * len(xl),
                                        coloraxis='coloraxis2',
                                        opacity=0,
                                        cmin=0, cmax=max_vals['last_bin'],
                                        ),
                            showlegend=False
                            )

    # strip left zero padding for better display
    ticktexts = [f"{int(pp.split('_')[0])}_{int(pp.split('_')[1])}"
                 for pp in amplitudes.columns.to_list()]
    _ = fig.update_xaxes(dict(scaleanchor=f'y',
                              constrain='domain',
                              range=[0, amplitudes.columns.size], fixedrange=True,
                              title_text='context pairs',
                              tickmode='array',
                              tickvals=np.arange(amplitudes.columns.size) + 0.5,
                              ticktext=ticktexts))

    _ = fig.update_yaxes(dict(title=dict(text=f'{picked_id}<br>probes'),
                              constrain='domain',
                              range=[0, amplitudes.index.size], fixedrange=True,
                              tickmode='array',
                              tickvals=np.arange(amplitudes.index.size) + 0.5,
                              ticktext=amplitudes.index.to_list()))

    # set the positions of the colorbars
    fig.update_layout(coloraxis=dict(colorscale=cscales['integral'],
                                     colorbar=dict(
                                         thickness=10, len=0.6,
                                         title_text='integral',
                                         title_side='right',
                                         tickangle=-90,
                                         xanchor='left', x=1)
                                     ),
                      coloraxis2=dict(colorscale=cscales['last_bin'],
                                      colorbar=dict(
                                          thickness=10, len=0.6,
                                          title_text='last_bin',
                                          title_side='right',
                                          tickangle=-90,
                                          xanchor='left', x=1.1)
                                      )
                      )

    return fig


if __name__ == '__main__':
    # for developing and debugging
    print('this should not be runned when importing')

    import pathlib as pl
    from configparser import ConfigParser

    import joblib as jl
    import pandas as pd

    from src.root_path import config_path

    config = ConfigParser()
    config.read_file(open(config_path / 'settings.ini'))
    meta = {'reliability': 0.1,  # r value
            'smoothing_window': 0,  # ms
            'raster_fs': 30,
            'montecarlo': 1000,
            'zscore': True,
            'stim_type': 'permutations'}


    cellid, contexts, probes = 'TNC006a-07-1', (2, 10), 2 # well-behaved example

    t_statistic = pl.Path(config['paths']['analysis_cache']) / f'220310_ctx_mod_metric_DF_tstat_cluster_mass_BS'
    df = jl.load(t_statistic)
    to_plot = df.query("metric in ['integral', 'last_bin'] and mult_comp_corr == 'bf_cp' and source == 'real' and "
                       "cluster_threshold == 0.05")
    tile = plot_tiling(cellid, to_plot)
    tile.show()

    # fig = make_subplots(1,4)
    #
    # raster = plot_raw_pair(cellid, contexts, probes, type='raster')
    # psth = plot_raw_pair(cellid, contexts, probes, type='psth')
    # quant0 = plot_time_ser_quant(cellid, contexts, probes, source='real',
    #                              multiple_comparisons_axis=[1,2], consecutive=0, cluster_threshold=0.05,
    #                              fn_name='t_statistic')
    # quant1 = plot_time_ser_quant(cellid, contexts, probes, source='real',
    #                              multiple_comparisons_axis=[1,2], consecutive=0, cluster_threshold=0.05,
    #                              fn_name='big_shuff', meta={'montecarlo': 11000})
    #
    # fig.add_traces(raster['data'],rows=[1]*len(raster['data']),cols=[1]*len(raster['data']))
    # fig.add_traces(psth['data'],rows=[1]*len(psth['data']),cols=[2]*len(psth['data']))
    # fig.add_traces(quant0['data'],rows=[1]*len(quant0['data']),cols=[3]*len(quant0['data']))
    # fig.add_traces(quant1['data'],rows=[1]*len(quant1['data']),cols=[4]*len(quant1['data']))
    #
    # fig.show()