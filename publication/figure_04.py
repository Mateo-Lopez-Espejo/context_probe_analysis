from itertools import combinations, product

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from webcolors import name_to_hex

from src.data.rasters import load_site_formated_raster
from src.metrics.simple_diff import ctx_effects_as_DF

from src.data.diagonalization import load_site_dense_raster
from src.visualization.palette import add_opacity, A1_COLOR, PEG_COLOR

from src.visualization.interactive import plot_site_coverages, plot_eg_diag, \
    plot_simple_psths

from publication.diagonalization_dataframe import accuracy_df
from publication.globals import raster_meta

# good looking examples
eg_site = 'ARM021b'
# The second neuron is the same used in fig 1 and 2
eg_neurons = [f'{eg_site}-{ci}' for ci in ['43-8', '36-8']]
eg_probes = [3, 4]
eg_ctxs = [1, 2, 3]
eg_ctx_pairs = list(combinations(eg_ctxs, 2))
eg_times = [5]

# load cache of sigle cell rasters for this example site
if load_site_formated_raster.check_call_in_cache(eg_site, **raster_meta):
    raster, cellids = load_site_formated_raster(eg_site, **raster_meta)
    print(f'####\n'
          f'found and loaded cache for {eg_site}')
else:
    print(f"cant load load_site_formated_raster with {raster_meta}."
          f"\n this should be cached, why is it failing? ")

# also loads full raster including context section for the graphical abstract
if load_site_formated_raster.check_call_in_cache(eg_site, part='all',
                                                 **raster_meta):
    fullraster, _ = load_site_formated_raster(eg_site, part='all',
                                              **raster_meta)
    print(f'####\n'
          f'found and loaded cache for {eg_site}')
else:
    print(f"cant load load_site_formated_raster with {raster_meta}."
          f"\n this should be cached, why is it failing? ")


def plot_diagonalization_geometric_example():
    """
    Figure 4 panels A and D. Shows the snapshot of two neuron activity as a 2d
    state space of a time point. It displays the activity associated to 2
    different probes after 3 contexts, showing the trial mean response and the
    confidence interval of the single trial variation.

    Returns: Plotly Figure

    """
    dense_raster, _ = load_site_dense_raster(eg_site, **raster_meta)
    rep, chn, ctx, prb, tme = raster.shape
    # smart index to select all reps for 2 neurons, all contexts,
    # two probes and one time point
    idxr = np.ix_(np.arange(rep), [cellids.index(cid) for cid in eg_neurons],
                  eg_ctxs, np.asarray(eg_probes) - 1, eg_times)

    fig = plot_eg_diag([raster, dense_raster], idxr, trial_mode='ellipse',
                       n_std=1, jitter=0.3, orientation='v')

    fig.update_layout(height=96 * 5, width=96 * 6,
                      margin=dict(l=10, r=10, t=10, b=10), showlegend=True)

    # axis labels
    _ = fig.update_xaxes(
        title=dict(text=f'neuron 1 activity (AU)<br>{eg_neurons[0]}',
                   standoff=0), row=2, col=1)
    _ = fig.update_yaxes(
        title=dict(text=f'neuron 2 activity (AU)<br>{eg_neurons[1]}',
                   standoff=0))
    fig.update_xaxes(constrain='domain', range=[-1.5, 6])
    fig.update_yaxes(constrain='domain', range=[-1.4, 3.5])

    return fig


def plot_example_coverage_with_diagonalization():
    """
    Figure 4 panels B and E. Coverage of context-pair probe combinations,
    similar to the ones in figure 3, however using a simplified calculation of
    the context effect amplitude that ignores the cluster mass significance
    calculation. This approach showcase the effect amplitude being equalized
    across all neurons in a site after the densification by diagonalization.
    This particular example shows two neurons sparse coverages and the dense
    coverage, which is equal for all neurons in the site


    Returns: Plolty figure

    """
    dense_raster, _ = load_site_dense_raster(eg_site, **raster_meta)

    sparse_df = ctx_effects_as_DF(raster, cellids, raster_meta['raster_fs'],
                                  abs=False).query(
        f"chunk == 'Full' and id in {eg_neurons}")  # simple diff

    # hack appende dense converage
    dense_df = ctx_effects_as_DF(dense_raster, cellids,
                                 raster_meta['raster_fs'], abs=False).query(
        f"chunk == 'Full' and id == '{eg_neurons[0]}'")
    dense_df['id'] = 'dense'

    toplot_df = pd.concat((sparse_df, dense_df))

    fig = plot_site_coverages(toplot_df, has_neg=True)
    fig.update_layout(height=96 * 2, width=96 * 3)

    return fig


def _plot_acc_from_df(fnDF, part, code, nsounds, color='black',
                      showlegend=False):
    """
    Parses the decoder performance dataframe and generates a figure of time
    series traces showing decoder accuracy over time for individual sites and
    for their average. Also shows the chance level

    Args:
        fnDF: Pandas DF
        part: str. Accuracy decoding 'context' or 'probe'
        code: str. Accuracy of 'sparse' or 'dense' code.
        nsounds: int. Accuracy for the 4 or 10 soundset experiments
        color: str. hex code or CSS named color for the lines
        showlegend: bool. shows legend for these lines

    Returns: Plotly figure

    """
    # transfroms color into hex if necesary
    if color[0] != '#':
        color = name_to_hex(color)

    fig = go.Figure()
    indf = fnDF.query(
        f"part=='{part}' and code=='{code}' and nsounds=={nsounds}")
    chance = indf.chance.unique()[0]
    arr = np.stack(indf.loc[:, 'accuracy'].values)

    # hardcoded 0 to 1 timeframe since I only care for decoding acc during
    # probe period
    t = np.linspace(0, 1, arr.shape[1],
                    endpoint=True)

    # individual lines, damn you plotly for this verbose monster
    for l in arr:
        fig.add_trace(
            go.Scatter(x=t, y=l, mode='lines', opacity=0.3, showlegend=False,
                       line=dict(color=add_opacity(color, 0.3)), ))

    # mean of all lines
    fig.add_trace(go.Scatter(x=t, y=arr.mean(axis=0), mode='lines',
                             showlegend=showlegend, line=dict(color=color),
                             name=f'{part}_{code}'))

    # chance line
    fig.add_trace(go.Scatter(x=[t[0], t[-1]], y=[chance] * 2, mode='lines',
                             showlegend=showlegend,
                             line=dict(color=color, dash='dot'),
                             name=f'chance={chance:.2f}'))

    return fig


def plot_summary_decoder_accuracies():
    """
    Figure 4 panels C and F.shows the decoder accuracy over time at predicting
    the context or probe information from the original sparse, or the
    diagonalized dense site responses.
    Only considers experiments done with the 10 sound soundset.

    Returns: Plotly Figure

    """
    encodings = ['sparse', 'dense']
    parts = ['context', 'probe']
    colors = ['green', 'purple']

    fig = make_subplots(rows=2, cols=1, shared_xaxes='all', shared_yaxes='all',
                        vertical_spacing=0.02, horizontal_spacing=0.02)

    for cc, code in enumerate(encodings):
        showlegend = True if cc == 0 else False
        for part, color in zip(parts, colors):
            f = _plot_acc_from_df(accuracy_df, part, code, nsounds=10,
                                  color=color, showlegend=showlegend)['data']
            fig.add_traces(f, rows=[cc + 1] * len(f), cols=[1] * len(f))

    fig.update_yaxes(title=dict(text=f'sparse<br>accuracy', standoff=0), row=1,
                     col=1)
    fig.update_yaxes(title=dict(text=f'dense<br>accuracy', standoff=0), row=2,
                     col=1)
    fig.update_xaxes(title=dict(text='time from probe (s)', standoff=0), row=2,
                     col=1)

    fig.update_layout(height=96 * 5, width=96 * 4, template='simple_white',
                      margin={'t': 20, 'l': 10, 'b': 10, 'r': 10},
                      showlegend=False)

    return fig


def plot_decoder_analysis_by_nsounds_and_region():
    """
    Supplementary figure 3 panels A to D. shows the decoder accuracy over time
    at predicting the context information (but not the probe info) from the
    original sparse, or the diagonalized dense site responses. Sites are split
    by cortical field.
    Both experiments done with 4 and 10 sound soundset are used.

    Returns: Plotly Figure

    """

    encodings = ['sparse', 'dense']
    regions = ['A1', 'PEG']
    colors = [A1_COLOR, PEG_COLOR]
    nsounds = [4, 10]

    fig = make_subplots(rows=2, cols=2, shared_xaxes='all', shared_yaxes='all',
                        subplot_titles=[f"{e} {n} sounds" for e, n in
                                        product(encodings, nsounds)],
                        vertical_spacing=0.02, horizontal_spacing=0.02)

    ii = 0
    for rr, code in enumerate(encodings):
        for cc, ns in enumerate(nsounds):
            row = rr + 1
            col = (cc % 2) + 1
            showlegend = True if ii == 0 else False
            ii += 1
            for region, color in zip(regions, colors):
                f = _plot_acc_from_df(
                    accuracy_df.query(f"region == '{region}'"), 'context',
                    code, nsounds=ns, color=color, showlegend=showlegend)[
                    'data']
                fig.add_traces(f, rows=[row] * len(f), cols=[col] * len(f))

    fig.update_yaxes(title=dict(text=f'accuracy', standoff=0), row=2, col=1)
    fig.update_xaxes(title=dict(text='time from probe (s)', standoff=0), row=2,
                     col=1)

    fig.update_layout(height=96 * 5 * 0.7, width=96 * 8 * 0.7,
                      template='simple_white',
                      margin={'t': 20, 'l': 10, 'b': 10, 'r': 10})

    return fig


def plot_simple_psht_with_avg():
    """
    Graphical abstract, left side panel. Simple PSTH of two context following
    one probe with the average probe response. No error shade and smoothed for
    simplicity and clarity

    Returns: Plotly figure

    """
    fig = plot_simple_psths([fullraster], cellids, eg_neurons, [1, 2], [3],
                            part='all', avg='visible')
    fig.add_vline(eg_times[0] * 1 / 20, line_color='gray', line_dash='dot',
                  line_width=2, opacity=1)
    fig.add_vline(0, line_color='black', line_dash='dot', line_width=2,
                  opacity=1)
    fig.update_layout(height=96 * 4, width=96 * 5)

    return fig
