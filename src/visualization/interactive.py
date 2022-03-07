import itertools as itt

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from webcolors import hex_to_rgb

from src.data.rasters import load_site_formated_raster
from src.metrics.consolidated_dprimes import single_cell_dprimes_cluster_mass
from src.metrics.consolidated_mean_diff import single_cell_mean_diff_cluster_mass
from src.metrics.consolidated_tstat import single_cell_tstat_cluster_mass
from src.metrics.significance import _significance
from src.visualization.fancy_plots import squarefy
from src.visualization.palette import *


def plot_psth_pair(cellid, contexts, probe):
    ctx_pair = contexts
    prb_idx = probe - 1

    site_raster, goodcellse = load_site_formated_raster(cellid[:7], part='all', smoothing_window=50)
    eg_raster = site_raster[:, goodcellse.index(cellid), :, prb_idx, :]

    # rotation of colors for the silence + 4 sound examples
    colors = [Grey, Yellow, Red, Teal, Brown]

    fig = go.Figure()
    for cxt_idx in ctx_pair:
        nsamps = eg_raster.shape[-1]
        time = np.linspace(-1, 1, nsamps)
        mean_resp = np.mean(eg_raster[:, cxt_idx, :], axis=0)
        std_resp = np.std(eg_raster[:, cxt_idx, :], axis=0)

        halfs = [np.s_[:int(nsamps / 2)], np.s_[int(nsamps / 2):]]
        part_color = [colors[cxt_idx % len(colors)], colors[prb_idx % len(colors)]]

        for nn, (half, color) in enumerate(zip(halfs, part_color)):

            x, y = squarefy(time[half], mean_resp[half])
            _, ystd = squarefy(time[half], std_resp[half])

            # off set half a bin to the left
            halfbin = np.mean(np.diff(time)) / 2
            x -= halfbin
            y -= halfbin
            ystd -= halfbin

            if nn == 0:
                # ax.fill_between(x, y-ystd, y+ystd, color=color, alpha=0.5)
                _ = fig.add_trace(go.Scatter(x=x, y=y + ystd, mode='lines', line_color=color, line_width=1))
                _ = fig.add_trace(go.Scatter(x=x, y=y - ystd, mode='lines', line_color=color, line_width=1,
                                             fill='tonexty'))

            else:
                # to set a transparent fillcolor changes the 'rgb(x,y,z)' into 'rgba(x,y,z,a)'
                rgb = hex_to_rgb(part_color[0])  # tupple
                fill_opacity = 0.5
                rgba = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {fill_opacity})'

                _ = fig.add_trace(go.Scatter(x=x, y=y + ystd, mode='lines', line_color=color, line_width=1))
                _ = fig.add_trace(go.Scatter(x=x, y=y - ystd, mode='lines', line_color=color, line_width=1,
                                             fill='tonexty', fillcolor=rgba))

            # set the mean lines second so they lie on top of the colored areas
            _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color=color, line_width=3))

    _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1)

    _ = fig.update_xaxes(title_text='time from probe onset (s)', title_standoff=0)
    _ = fig.update_yaxes(title_text='firing rate (z-score)', title_standoff=0)

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
               't_statistic':single_cell_tstat_cluster_mass}

    fn = fn_dict[fn_name]
    meta.update(defaults_meta)

    if fn.check_call_in_cache(cellid[:7], contexts='all', probes='all', cluster_threshold=float(cluster_threshold), meta=meta):
        dprime, pval_quantiles, goodcells, shuffled_eg = fn(
            cellid[:7], contexts='all', probes='all', cluster_threshold=float(cluster_threshold), meta=meta
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
        n_comparisons = 1
    else:
        n_comparisons = np.prod(np.asarray(pvalue.shape)[np.asarray(multiple_comparisons_axis)])

    corr_alpha = f"{alpha / n_comparisons:.5f}"


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
    if corr_alpha in pval_quantiles.keys():
        CI = pval_quantiles[corr_alpha][cell_idx, pair_idx, prb_idx, 0]
    else:
        print('undefined quantiles for ')
    CT = pval_quantiles['clusters'][cell_idx, pair_idx, prb_idx, :] * flip

    if 't-threshold' in pval_quantiles:
        # dinamically defined threshold for t test. depends on degrees of fredom i.e. reps
        CTT = pval_quantiles['t-threshold']
        print(f'using t-score threshold for sample-alpha {cluster_threshold} -> t = {CTT}')
    else:
        CTT = cluster_threshold

    SIG = significance[cell_idx, pair_idx, prb_idx, :]

    signif_mask = SIG > 0
    t = np.linspace(0, DP.shape[-1] / meta['raster_fs'], DP.shape[-1], endpoint=False)

    # calculates center of mass and integral
    significant_abs_mass_center = np.sum(np.abs(DP[signif_mask]) * t[signif_mask]) / np.sum(np.abs(DP[signif_mask]))
    # if np.isnan(significant_abs_mass_center): significant_abs_mass_center = 0
    significant_abs_sum = np.sum(np.abs(DP[signif_mask])) * np.mean(np.diff(t))
    print(f"integral: {significant_abs_sum * 1000:.2f} d'*ms")
    print(f'center of mass: {significant_abs_mass_center * 1000:.2f} ms')

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
    if not np.isnan(significant_abs_mass_center):
        ytop = DP[np.abs(t-significant_abs_mass_center).argmin()]
        _ = fig.add_trace(go.Scatter(x=[significant_abs_mass_center]*2, y=[0, ytop], mode='lines',
                                     line=dict(color=Purple, width=4)))

    # formats axis, legend and so on.
    _ = fig.update_xaxes(title=dict(text='time from probe onset (s)', standoff=0))

    _ = fig.update_yaxes(title=dict(text="contexts d'", standoff=0))

    return fig

def plot_neuron_tiling(picked_neuron, df):
    # turns long format data into an array with dimension Probe * context_pair
    to_pivot = df.loc[df['id'] == picked_neuron, :]
    val_df = to_pivot.pivot_table(index=['metric', 'probe'], columns=['context_pair'], values='value')

    cscales = {'amplitude': pc.make_colorscale(['#000000', Green]),
               'duration': pc.make_colorscale(['#000000', Purple])}
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

    amp_color = color_df.loc[('amplitude'), :].values
    dur_color = color_df.loc[('duration'), :].values

    amplitudes = val_df.loc[('amplitude'), :]
    durations = val_df.loc[('duration'), :]

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
                                        cmin=0, cmax=max_vals['amplitude'],
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
                                        cmin=0, cmax=max_vals['duration'],
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

    _ = fig.update_yaxes(dict(title=dict(text=f'{picked_neuron}<br>probes'),
                              constrain='domain',
                              range=[0, amplitudes.index.size], fixedrange=True,
                              tickmode='array',
                              tickvals=np.arange(amplitudes.index.size) + 0.5,
                              ticktext=amplitudes.index.to_list()))

    # set the positions of the colorbars
    fig.update_layout(coloraxis=dict(colorscale=cscales['amplitude'],
                                     colorbar=dict(
                                         thickness=10, len=0.6,
                                         title_text='amplitude',
                                         title_side='right',
                                         tickangle=-90,
                                         xanchor='left', x=1)
                                     ),
                      coloraxis2=dict(colorscale=cscales['duration'],
                                      colorbar=dict(
                                          thickness=10, len=0.6,
                                          title_text='duration',
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
    t_statistic = pl.Path(config['paths']['analysis_cache']) / f'220303_ctx_mod_metric_DF_tstat_cluster_mass'
    df = jl.load(t_statistic)

    def format_dataframe(DF):
        ff_analylis = DF.analysis.isin(['SC'])
        ff_badsites = ~DF.siteid.isin(['TNC010a'])
        mask = ff_analylis & ff_badsites

        if 'cluster_threshold' not in DF.columns:
            DF['cluster_threshold'] = 0

        good_cols = ['source', 'mult_comp_corr', 'cluster_threshold', 'region', 'siteid', 'cellid', 'context_pair',
                     'probe', 'metric', 'value']
        filtered = DF.loc[mask, good_cols]

        filtered['probe'] = [int(p) for p in filtered['probe']]
        filtered['context_pair'] = [f"{int(cp.split('_')[0]):02d}_{int(cp.split('_')[1]):02d}"
                                    for cp in filtered['context_pair']]

        # rename metrics and analysis for ease of ploting
        filtered['metric'] = filtered['metric'].replace({'significant_abs_mass_center': 'duration',
                                                         'significant_abs_sum': 'amplitude'})

        filtered['id'] = filtered['cellid'].fillna(value=filtered['siteid'])
        filtered = filtered.drop(columns=['cellid'])
        filtered.rename(columns={'siteid': 'site'}, inplace=True)

        filtered['value'] = filtered['value'].fillna(value=0)

        # permutation related preprocesing.
        # creates a new column relating probe with  context pairs
        ctx = np.asarray([row.split('_') for row in filtered.context_pair], dtype=int)
        prb = np.asarray(filtered.probe, dtype=int)

        silence = ctx == 0
        same = ctx == prb[:, None]
        different = np.logical_and(~silence, ~same)

        name_arr = np.full_like(ctx, np.nan, dtype=object)
        name_arr[silence] = 'silence'
        name_arr[same] = 'same'
        name_arr[different] = 'diff'
        comp_name_arr = np.apply_along_axis('_'.join, 1, name_arr)

        # swaps clasification names to not have repetitions i.e. diff_same == same_diff
        comp_name_arr[np.where(comp_name_arr == 'same_silence')] = 'silence_same'
        comp_name_arr[np.where(comp_name_arr == 'diff_silence')] = 'silence_diff'
        comp_name_arr[np.where(comp_name_arr == 'diff_same')] = 'same_diff'
        comp_name_arr[np.where(comp_name_arr == 'same_silence')] = 'silence_same'

        filtered['trans_pair'] = comp_name_arr

        # column specifying number of different sounds used
        nstim = filtered.groupby(['id']).agg(stim_count=('probe', lambda x: x.nunique()))
        filtered = pd.merge(filtered, nstim, on='id')

        return filtered

    cellid, contexts, probes = 'ARM021b-36-8', (0, 1), 3  # odd tail in bf_cp-2.0
    cellid, contexts, probes = 'CRD012b-13-1', (1, 4), 3  # big difference by cluster threshold
    cellid, contexts, probes = 'TNC006a-07-1', (2, 10), 2 # well behaved example

    # longDF = format_dataframe(df)
    # tile = plot_neuron_tiling(cellid, longDF)
    # tile.show()

    # psth = plot_psth_pair(cellid, contexts, probes)
    # psth.show()


    # fig = make_subplots(1,3)
    #
    # left = plot_time_ser_quant(cellid, contexts, probes,
    #                              multiple_comparisons_axis=[1,2], consecutive=0, cluster_threshold=2,
    #                              fn_name='dprime')
    #
    # center = plot_time_ser_quant(cellid, contexts, probes,
    #                              multiple_comparisons_axis=[1,2], consecutive=0, cluster_threshold=2,
    #                              fn_name='mean_difference')
    #
    # right = plot_time_ser_quant(cellid, contexts, probes,
    #                              multiple_comparisons_axis=[1,2], consecutive=0, cluster_threshold=0.05,
    #                              fn_name='t_statistic')
    #
    # fig.add_traces(left['data'],rows=[1]*len(left['data']),cols=[1]*len(left['data']))
    # fig.add_traces(center['data'],rows=[1]*len(center['data']),cols=[2]*len(center['data']))
    # fig.add_traces(right['data'],rows=[1]*len(right['data']),cols=[3]*len(right['data']))
    #
    # fig.show()
    picked_eg = {'points': [{'curveNumber': 21, 'pointNumber': 7244, 'pointIndex': 7244, 'x': 185.252606486073, 'y': 3215.425125645705, 'hovertext': 'TNC014a-22-2', 'bbox': {'x0': 302.49, 'x1': 304.49, 'y0': 163.64, 'y1': 165.64}, 'customdata': ['00_03', 2]}]}
    picked_eg = {'points': [
        {'curveNumber': 20, 'pointNumber': 15856, 'pointIndex': 15856, 'x': 838.4125227037579, 'y': 1008.7867703423386,
         'hovertext': 'TNC013a-46-3', 'bbox': {'x0': 873.57, 'x1': 875.57, 'y0': 299.82, 'y1': 301.82},
         'customdata': ['07_10', 6]}]}


    cellid = picked_eg['points'][0]['hovertext']
    contexts = [int(ss) for ss in picked_eg['points'][0]['customdata'][0].split('_')]
    probes = picked_eg['points'][0]['customdata'][1]

    fig = make_subplots(1, 2)

    psth = plot_psth_pair(cellid, contexts, probes)
    quant_diff = plot_time_ser_quant(cellid, contexts, probes,
                                     multiple_comparisons_axis=[1, 2], consecutive=0, cluster_threshold=0.05,
                                     fn_name='t_statistic', meta=meta)

    fig.add_traces(psth['data'], rows=[1] * len(psth['data']), cols=[1] * len(psth['data']))
    fig.add_traces(quant_diff['data'], rows=[1] * len(quant_diff['data']), cols=[2] * len(quant_diff['data']))