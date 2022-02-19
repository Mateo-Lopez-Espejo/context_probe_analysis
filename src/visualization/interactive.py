import itertools as itt

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from webcolors import hex_to_rgb

from src.data.rasters import load_site_formated_raster
from src.metrics.consolidated_dprimes import single_cell_dprimes
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
        part_color = [colors[cxt_idx % len(colors)], colors[prb_idx & len(colors)]]

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


def plot_dprime_quant(cellid, contexts, probe, source='real',
                      multiple_comparisons_axis=[3], consecutive=3):
    meta = {'alpha': 0.05,
            'montecarlo': 1000,
            'raster_fs': 30,
            'reliability': 0.1,
            'smoothing_window': 0,
            'stim_type': 'permutations',
            'zscore': True}
    dprime, shuff_dprime_quantiles, goodcells, shuffled_eg = single_cell_dprimes(cellid[:7], contexts='all',
                                                                                 probes='all', meta=meta)

    if source == 'real':
        dprime = dprime
    elif source == 'shuffled_eg':
        dprime = shuffled_eg['dprime']

    significance, confidence_interval = _significance(dprime, shuff_dprime_quantiles,
                                                      multiple_comparisons_axis=multiple_comparisons_axis,
                                                      consecutive=consecutive,
                                                      alpha=meta['alpha'])
    cell_idx = goodcells.index(cellid) if len(cellid) > 7 else 0

    pair_idx = [f'{t0}_{t1}' for t0, t1 in itt.combinations(range(dprime.shape[2] + 1), 2)].index(
        f'{contexts[0]}_{contexts[1]}')

    prb_idx = probe - 1

    DP = dprime[cell_idx, pair_idx, prb_idx, :] * -1
    CI = confidence_interval[:, cell_idx, pair_idx, prb_idx, :] * -1
    SIG = significance[cell_idx, pair_idx, prb_idx, :]
    raster_fs = meta['raster_fs']

    signif_mask = SIG > 0
    t = np.linspace(0, DP.shape[-1] / raster_fs, DP.shape[-1], endpoint=False)

    # calculates center of mass and integral
    significant_abs_mass_center = np.sum(np.abs(DP[signif_mask]) * t[signif_mask]) / np.sum(np.abs(DP[signif_mask]))
    significant_abs_sum = np.sum(np.abs(DP[signif_mask])) * np.mean(np.diff(t))
    print(f"integral: {significant_abs_sum * 1000:.2f} d'*ms")
    print(f'center of mass: {significant_abs_mass_center * 1000:.2f} ms')

    fig = go.Figure()
    # plots dprime
    tt, mmdd = squarefy(t, DP)
    _ = fig.add_trace(go.Scatter(x=tt, y=mmdd, mode='lines', line_color='black', line_width=3))

    # significance confidence interval
    _, CCII = squarefy(t, CI.T)
    _ = fig.add_trace(go.Scatter(x=tt, y=CCII[:, 0], mode='lines', line_color='gray', line_width=1))
    _ = fig.add_trace(go.Scatter(x=tt, y=CCII[:, 1], mode='lines', line_color='gray', line_width=1,
                                 fill='tonexty'))

    # significant area under the curve
    # little hack to add gaps into the area, set d' value to zero where no significance
    _, smm = squarefy(t, signif_mask)
    wmmdd = np.where(smm, mmdd, 0)
    rgb = hex_to_rgb(Green)
    rgba = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.5)'

    _ = fig.add_trace(go.Scatter(x=tt, y=wmmdd, mode='none',
                                 fill='tozeroy', fillcolor=rgba))

    # center of mass indication
    _ = fig.add_vline(significant_abs_mass_center, line=dict(color=Purple, dash='dash', width=3))

    # general plot formating
    _ = fig.add_hline(0, line=dict(dash='dot', width=2, color='black'))
    # formats axis, legend and so on.

    _ = fig.update_xaxes(title=dict(text='time from probe onset (s)', standoff=0))

    _ = fig.update_yaxes(title=dict(text="contexts d'", standoff=0))

    return fig


def plot_neuron_tiling(picked_neuron, df):
    # turns long format data into an array with dimension Probe * context_pair
    to_pivot = df.loc[df['id'] == picked_neuron, :]
    val_df = to_pivot.pivot_table(index=['metric', 'probe'], columns=['context_pair'], values='value')

    cscales = {"integral (d'*ms)": pc.make_colorscale(['#000000', Green]),
               "center of mass (ms)": pc.make_colorscale(['#000000', Purple])}
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

    amp_color = color_df.loc[("integral (d'*ms)"), :].values
    dur_color = color_df.loc[('center of mass (ms)'), :].values

    amplitudes = val_df.loc[("integral (d'*ms)"), :]
    durations = val_df.loc[('center of mass (ms)'), :]

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
                                        cmin=0, cmax=max_vals["integral (d'*ms)"],
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
                                        cmin=0, cmax=max_vals["center of mass (ms)"],
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
    fig.update_layout(coloraxis=dict(colorscale=cscales["integral (d'*ms)"],
                                     colorbar=dict(
                                         thickness=10, len=0.6,
                                         title_text="amplitude (d'*ms)",
                                         title_side='right',
                                         tickangle=-90,
                                         xanchor='left', x=1)
                                     ),
                      coloraxis2=dict(colorscale=cscales["center of mass (ms)"],
                                      colorbar=dict(
                                          thickness=10, len=0.6,
                                          title_text="duration (ms)",
                                          title_side='right',
                                          tickangle=-90,
                                          xanchor='left', x=1.1)
                                      )
                      )

    return fig


if __name__ == '__main__':
    # for developing and debugging

    import pathlib as pl
    from configparser import ConfigParser

    import joblib as jl
    import pandas as pd

    from src.root_path import config_path

    config = ConfigParser()
    config.read_file(open(config_path / 'settings.ini'))
    meta = {'alpha': 0.05,
            'montecarlo': 1000,
            'raster_fs': 30,
            'reliability': 0.1,
            'smoothing_window': 0,
            'stim_type': 'permutations',
            'zscore': True}
    summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220214_ctx_mod_metric_DF'
    df = jl.load(summary_DF_file)


    def format_dataframe(DF):
        ff_analylis = DF.analysis.isin(['SC', 'fdPCA'])
        ff_corr = DF.mult_comp_corr == 'consecutive_3'

        good_cols = ['source', 'analysis', 'mult_comp_corr', 'region', 'siteid', 'cellid', 'context_pair',
                     'probe', 'metric', 'value']
        filtered = DF.loc[ff_analylis & ff_corr, good_cols]

        filtered['probe'] = [int(p) for p in filtered['probe']]
        filtered['context_pair'] = [f"{int(cp.split('_')[0]):02d}_{int(cp.split('_')[1]):02d}"
                                    for cp in filtered['context_pair']]

        # rename metrics and analysis for ease of ploting
        filtered['metric'] = filtered['metric'].replace({'significant_abs_mass_center': 'center of mass (ms)',
                                                         'significant_abs_mean': "mean d'",
                                                         'significant_abs_sum': "integral (d'*ms)"})
        filtered['analysis'] = filtered['analysis'].replace({'SC': 'single cell',
                                                             'fdPCA': 'population',
                                                             'pdPCA': 'probewise pop',
                                                             'LDA': 'pop ceiling'})

        filtered['id'] = filtered['cellid'].fillna(value=filtered['siteid'])
        filtered = filtered.drop(columns=['cellid', 'siteid'])

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

        nstim = filtered.groupby(['id']).agg(stim_count=('probe', lambda x: x.nunique()))
        filtered = pd.merge(filtered, nstim, on='id')

        return filtered


    longDF = format_dataframe(df)

    cellid, contexts, probes = 'ARM021b-36-8', (0, 1), 3  # paper example
    cellid, contexts, probes = 'TNC010a-40-1', (2, 10), 4

    tile = plot_neuron_tiling(cellid, longDF)
    tile.show()

    psth = plot_psth_pair(cellid, contexts, probes)
    psth.show()

    dprm_fig = plot_dprime_quant(cellid, contexts, probes)
    dprm_fig.show()
