"""
Figure 1 panels A, C, D, E, F, G, K
"""
from math import pi
import itertools as itt

import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.visualization.interactive import plot_raw_pair, plot_time_ser_quant, \
    plot_simple_quant
from src.visualization.palette import *

# waveforms placeholder
n_samps = 100
dummy_wave = np.sin(np.linspace(0, pi * 4, n_samps)) * 0.25
waves = [np.zeros(n_samps)] + [dummy_wave, ] * 5 + [np.zeros(n_samps)]

eg_probe = 4
colors = FOURCOLOR


def plot_sequence_order():
    """
    Figure 1 Panel A. Example of sequence order for 4 different sounds such
    that all sounds are presented after all other sounds, silence
    and themselves. In the Paper figure I ended up using good-looking snippets
    of me making random noises. good enough for a cartoon I believe.
    May the reader forgive me, and  replace the placeholder sinewaves for
    whatever their heart desires.

    Returns: Plotly figure
    """

    # hardcodded order determined with Knuth's X algorithm.
    sequences = np.asarray(
        [[0, 1, 3, 2, 4, 4], [0, 3, 4, 1, 1, 2], [0, 4, 2, 3, 3, 1],
         [0, 2, 2, 1, 4, 3]])

    xbox = np.asarray([0, 1, 1, 0, 0])
    ybox = (np.asarray([0, 0, 1, 1, 0]) - 0.5) * 0.75

    egbox_height = 0.8

    fig = go.Figure()

    for ss, seq in enumerate(sequences):
        for ww, wave_idx in enumerate(seq):
            color = colors[wave_idx]
            if ww > 0:
                # Colored boxes except silence
                _ = fig.add_trace(
                    go.Scatter(x=xbox + ww, y=ybox + ss, fill='toself',
                               mode='lines', line=dict(width=1, color='gray'),
                               fillcolor=color, showlegend=False))

            # wave form plots
            x = np.linspace(0, 1, n_samps) + ww
            y = waves[wave_idx] + ss
            _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                         line=dict(color='black', width=1),
                                         showlegend=False, ))

    # Add e.g. dotted boxes, ensure are the last to be drawn so they are on top
    for ss, seq in enumerate(sequences):
        for ww, wave_idx in enumerate(seq):
            if wave_idx == eg_probe:
                x0 = ww - 1
                y0 = ss - egbox_height / 2
                xd, yd = 2, egbox_height  # 2 seconds width, 2*norm wave
                x = [x0, x0, x0 + xd, x0 + xd, x0]
                y = [y0, y0 + yd, y0 + yd, y0, y0]
                _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                             line=dict(color='black', width=2,
                                                       dash='dot'),
                                             showlegend=False, ))

    # test show
    _ = fig.update_xaxes(title_text='time (s)', title_standoff=0,
                         range=[-0.1, 6.1])
    _ = fig.update_yaxes(tickmode='array', tickvals=list(range(4)),
                         ticktext=[f'Seq.{i + 1}' for i in range(4)], ticks='',
                         showline=False)
    fig.update_layout(width=96 * 3, height=96 * 1.5,
                      margin={'l': 10, 'r': 10, 't': 10, 'b': 10, },
                      template='simple_white')

    return fig


def plot_selected_sound():
    """
    Figure 1 panel C. Selecting one of the 4 sounds, we can see it follows all
    oterh sounds, silence and itself. Here with dummy waveforms as in panel A.
    Returns: Plotly figure

    """

    eg_probe = 4

    colors = FOURCOLOR

    fig = go.Figure()
    xbox = np.asarray([0, 1, 1, 0, 0])
    ybox = (np.asarray([0, 0, 1, 1, 0]) - 0.5) * 0.75

    for ww, (wave, color) in enumerate(zip(waves, colors)):
        # context box
        if ww > 0:  # omits silence box,
            _ = fig.add_trace(
                go.Scatter(x=xbox - 1, y=ybox + ww, fill='toself',
                           mode='lines', line=dict(width=1, color='gray'),
                           fillcolor=color, showlegend=False))
        # contex wave
        x = np.linspace(-1, 0,
                        n_samps)  # sum to offset to center, insline with sequences
        y = wave + ww
        _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                     line=dict(color='black', width=1, ),
                                     showlegend=False))

        # probe box
        _ = fig.add_trace(
            go.Scatter(x=xbox, y=ybox + ww, fill='toself', mode='lines',
                       line=dict(width=1, color='gray'),
                       fillcolor=colors[eg_probe], showlegend=False))
        # probe wave
        x = np.linspace(0, 1, n_samps)
        y = waves[eg_probe] + ww
        _ = fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                     line=dict(color='black', width=1, ),
                                     showlegend=False))
        # ax.plot(x, y, colors[prb_idx])

        # context type text
        if ww == 0:
            type_text = 'silence'
        elif ww == eg_probe:
            type_text = 'same'
        else:
            type_text = 'different'

        _ = fig.add_trace(
            go.Scatter(x=[-1.1], y=[ww], mode='text', text=[type_text],
                       textposition='middle left', textfont_size=11,
                       showlegend=False))

    _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot',
                      opacity=1, )
    # context and probe text
    _ = fig.add_trace(go.Scatter(x=[-0.2, 0.2], y=[-1, -1], mode='text',
                                 text=['<b>Context</b>', '<b>Probe</b>'],
                                 textposition=['middle left', 'middle right'],
                                 textfont_size=12, showlegend=False))

    # test show
    _ = fig.update_layout(width=96 * 2.5, height=96 * 1.5,
                          margin={'l': 10, 'r': 10, 't': 10, 'b': 10, },
                          template='simple_white',
                          xaxis=dict(range=[-2, 1.5], visible=False),
                          yaxis=dict(visible=False))

    return fig


def plot_example_quantifications():
    """
    Figure 1 Panels D to K. Example rasters, PSTHs, signficiance calculations
    and context metric quantification. These two examples were handpicked for
    their good looks, yet they are representative, I asure you.
    Returns:

    """
    # examples as a cellid, a list of context and a probe
    examples = [('ARM021b-36-8', [1, 3], 4), ('ARM021b-36-8', [0, 1], 3), ]
    fig = make_subplots(2, 4, shared_xaxes='columns', shared_yaxes='columns',
                        column_widths=[2 / 6, 2 / 6, 1 / 6, 1 / 6],
                        horizontal_spacing=0.05, vertical_spacing=0.05,
                        specs=[[{}, {}, {"secondary_y": True}, {}],
                               [{}, {}, {"secondary_y": True}, {}]])
    for ee, (cellid, ctx_pair, probe) in enumerate(examples):
        # raster
        f = plot_raw_pair(cellid, ctx_pair, probe, mode='raster',
                          raster_fs=100)
        f.update_traces(line=dict(width=1), showlegend=False,
                        marker=dict(size=2, opacity=0.8, line=dict(width=0, )))

        t = f['data']
        fig.add_traces(t, rows=[ee + 1] * len(t), cols=[1] * len(t))

        # PSTH
        f = plot_raw_pair(cellid, ctx_pair, probe, raster_fs=20, simplify=True)
        f.update_traces(line=dict(width=1))
        if ee != 0:
            f.update_traces(showlegend=False)

        t = f['data']
        fig.add_traces(t, rows=[ee + 1] * len(t), cols=[2] * len(t))

        # significance
        f, main_traces, secondary_traces = plot_time_ser_quant(cellid,
            ctx_pair, probe, multiple_comparisons_axis=[1, 2],
            cluster_threshold=0.05, meta={'raster_fs': 20}, secondary_y=True,
            deltaFR=False, ignore_quant=True)
        for trace in main_traces + secondary_traces:
            trace.update(line_width=1, marker_size=7)
            if ee != 0:
                trace.update(showlegend=False)

        t = main_traces
        fig.add_traces(t, rows=[ee + 1] * len(t), cols=[3] * len(t))

        t = secondary_traces
        fig.add_traces(t, rows=[ee + 1] * len(t), cols=[3] * len(t),
                       secondary_ys=[True] * len(t))

        # quantification
        f = plot_simple_quant(cellid, ctx_pair, probe,
            multiple_comparisons_axis=[1, 2], cluster_threshold=0.05,
            meta={'raster_fs': 20}, )
        f.update_traces(line_width=1, marker_size=7)
        if ee != 0:
            f.update_traces(showlegend=False)

        t = f['data']
        fig.add_traces(t, rows=[ee + 1] * len(t), cols=[4] * len(t))

    # Axes label formating and other details.
    _ = fig.update_layout(template='simple_white',
                          margin=dict(l=10, r=10, t=10, b=10),
                          width=round(96 * 9), height=round(96 * 3),

                          # top raster
                          yaxis=dict(showticklabels=False, ticks=''),

                          # bottom raster
                          xaxis5=dict(title_text='time from probe onset (s)'),
                          yaxis6=dict(showticklabels=False, ticks='',
                                      title_text='trials'),

                          # bottom PSTH
                          xaxis6=dict(title_text='time from probe onset (s)'),
                          yaxis7=dict(title_text='firing rate (z-score)'),

                          # top significance
                          yaxis4=dict(matches='y9'),

                          # bottom significance
                          xaxis7=dict(title_text='time from probe onset (s)'),
                          yaxis8=dict(title_text='Difference (T-score)'),
                          yaxis9=dict(title=dict(text='Cluster Sum (T-score)',
                                                 font_color=AMPCOLOR)),

                          # bottom quantification
                          xaxis8=dict(title_text='time from probe onset (s)'),
                          yaxis10=dict(title_text='Difference (Z-score)'),

                          legend_font_size=9)

    _ = fig.update_xaxes(title=dict(font_size=10, standoff=0),
                         tickfont_size=9, )
    _ = fig.update_yaxes(title=dict(font_size=10, standoff=0),
                         tickfont_size=9, )

    # ranges  for rasterss
    for rr, cc in itt.product([1, 2], [1]):
        _ = fig.update_xaxes(range=[-1, 1], row=rr, col=cc)
        _ = fig.update_yaxes(range=[0, 40], row=rr, col=cc)

    # ranges  for psths
    for rr, cc in itt.product([1, 2], [2]):
        _ = fig.update_xaxes(range=[-1, 1], row=rr, col=cc)
        _ = fig.update_yaxes(autorange=True, row=rr, col=cc)

    # ranges for significance
    for rr, cc in itt.product([1, 2], [3]):
        _ = fig.update_xaxes(range=[0, 1], row=rr, col=cc)
        _ = fig.update_yaxes(range=[0, 13], row=rr, col=cc, secondary_y=False)
        _ = fig.update_yaxes(range=[0, 100], row=rr, col=cc, secondary_y=True)

    # ranges for quantification
    for rr, cc in itt.product([1, 2], [4]):
        _ = fig.update_xaxes(range=[0, 1], row=rr, col=cc)
        _ = fig.update_yaxes(range=[0, 5], row=rr, col=cc)

    # add vertical dashed lines at necesary probe onsets (x=0)
    panxy = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for rr, cc in panxy:
        _ = fig.add_vline(x=0, line_width=1, line_color='black',
                          line_dash='dot', opacity=1, row=rr, col=cc)

    # horizontal lines dividing different context trials in rates
    panxy = [(1, 1), (2, 1)]
    for rr, cc in panxy:
        trialn = 20
        _ = fig.add_hline(y=trialn - 0.5, line_width=1, line_color='gray',
                          line_dash='dash', opacity=0.5, row=rr, col=cc)

    return fig
