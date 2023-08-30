import numpy as np
import scipy.stats as sst
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from publication.globals import SUP_FIG3_WDF

from src.utils.tools import decimate_xy
from src.visualization.interactive import plot_raw_pair, plot_pupil_so_effects
from src.visualization.palette import *


def plot_exmaple_pupil_psths():
    """
    Supplementary figure 3, panels A and B. Shows an example neuron response to
    a probe after two different contexts, split for trials in which the pupil
    was small or large. Also shows the context effect as the delta firing rate
    split for large and small pupil.

    Note that the legend is covering the last panel, and this was solved post
    hoc exporting the figure as vector graphics (.svg) for styling.

    Returns: Plotly figure.

    """
    # first find an example, the following conditions have to be met
    # positive big first order effects (on full duration)
    # positive big second order effects (on chunk A)
    # big contextual effects

    tosort = SUP_FIG3_WDF.query(
        "small_fo > 0.1  and big_fo > 0.1"
        " and ((full_so <= -0.3) or (0.3 <= full_so))"
        " and mod_coeff_corrected_fo > 0"
        " and mod_coeff_corrected_so > 0"
        " and chunk == 'A'"
    ).loc[:,
             ['id', 'context_pair', 'probe', 'mod_coeff_corrected_fo',
              'mod_coeff_corrected_so', 'value']
             ].copy()

    tosort['ctx_mod_norm'] = tosort['value'] / tosort[
        'value'].max()
    tosort['euc_dist'] = np.linalg.norm(
        tosort.loc[
        :, ['mod_coeff_corrected_fo', 'mod_coeff_corrected_so', 'ctx_mod_norm']
        ].values, axis=1
    )

    tosort.sort_values(by='euc_dist', ascending=False, inplace=True,
                       ignore_index=True)

    ii = 0
    cellid = tosort.loc[ii, 'id']
    contexts = [int(cc) for cc in tosort.loc[ii, 'context_pair'].split('_')]
    probe = tosort.loc[ii, 'probe']

    fig = make_subplots(1, 3, shared_xaxes='all')

    # big pupil
    f = plot_raw_pair(cellid, contexts, probe, raster_fs=20, colors=TENCOLOR,
                      pupil='big', simplify=True, part='probe')
    f.update_traces(line=dict(width=1))
    f = f['data']
    fig.add_traces(f, cols=[1] * len(f), rows=[1] * len(f))

    # small pupil
    f = plot_raw_pair(cellid, contexts, probe, raster_fs=20, colors=TENCOLOR,
                      pupil='small', simplify=True, part='probe')
    f.update_traces(line=dict(dash='dot', width=1), showlegend=False)
    f = f['data']
    fig.add_traces(f, cols=[2] * len(f), rows=[1] * len(f))

    # delta firing rates
    f = plot_pupil_so_effects(cellid, contexts, probe, raster_fs=20)
    f.update_traces(line=dict(width=1))

    f = f['data']
    fig.add_traces(f, cols=[3] * len(f), rows=[1] * len(f))

    w, h = 4.5, 1.5
    fig.update_layout(template='simple_white',
                      width=96 * w, height=96 * h,
                      margin=dict(l=10, r=10, t=10, b=10),

                      yaxis_title_text='firin rate (Z-score)',
                      yaxis2=dict(matches='y', showticklabels=False),
                      yaxis3=dict(title_text='delta firing rate (Z-score)'),
                      legend=dict(xanchor='right', x=1,
                                  yanchor='top', y=1,
                                  font_size=9,
                                  title=dict(text='')),
                      )

    fig.update_xaxes(
        title=dict(text='time from probe onset (s)', font_size=10, standoff=0),
        tickfont_size=9)
    fig.update_yaxes(title=dict(font_size=10, standoff=0),
                     tickfont_size=9,
                     autorange=True)

    return fig


def plot_first_vs_second_order_effects():
    """
    Supplementary figure 3 panel C. Shows a scatter plot of  instances
    first order (firing rate) vs second order (context effects)
    pupil modulation index, split by 250ms time intervals. Displays the
    regression line between the MI and prints statistical tests for the
    regression being significantly different that zero (Wald test), and tests
    for the means of the first and second order MI being significantly
    different from zero (1 sample t-test). For clarity only a random subset
    of points is displayed, but all the data is used for the tests.
    Aditionally the data is filtered to remove unresponsive neurons and
    instances with reduced context effects.

    Returns: Plotly figure.

    """
    # statistical tests for the regression across all time chunks
    chunks = ['A', 'B', 'C', 'D']
    fig = make_subplots(1, 4, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=chunks)

    for coln, chunk in enumerate(chunks):

        # Filters out instances with low context independent firing rates, and
        # no asociated pupil effects, and with low pupil independent context
        # effects.
        toplot = SUP_FIG3_WDF.query(
            f"small_fo > 0.1 and big_fo > 0.1 "
            f"and ((full_so <= -0.3) or (0.3 <= full_so)) "
            f"and chunk == '{chunk}'"
        ).copy()

        x = toplot['mod_coeff_corrected_fo'].values
        y = toplot['mod_coeff_corrected_so'].values

        ####### statistical tests #######
        print(
            f'\n###########\n'
            f'chunk {chunk}, n_neuron={toplot.id.nunique()}, '
            f'n_instanceses={len(x)}'
        )
        print(f'mean MI: first_ord={x.mean()}, second_ord{y.mean()}')
        # marginal differece from zero
        for marg, vctr in zip(['FR_MI', 'ctx_MI'], [x, y]):
            out = sst.ttest_1samp(vctr, 0)
            print(f'{marg}: {out}')

        # correaltion
        print(f"first vs second order modulation index:"
              f"\n{sst.linregress(x, y)}")

        # main scatter
        xm, ym = decimate_xy(x, y, 2000, by_quantiles=False,
                             rng=np.random.default_rng(42))
        print(len(xm))
        _ = fig.add_trace(go.Scatter(x=xm, y=ym, mode='markers',
                                     marker=dict(size=1,
                                                 opacity=0.3,
                                                 color='black'),
                                     showlegend=False),
                          row=1, col=coln + 1)

        # interval for unit and regression
        interval = np.asarray([np.min([xm.min(), ym.min()]),
                               np.max([xm.max(), ym.max()])])

        # regression
        reg = sst.linregress(x, y)
        _ = fig.add_trace(
            go.Scatter(x=interval, y=interval * reg.slope + reg.intercept,
                       mode='lines',
                       line=dict(dash='solid',
                                 width=1,
                                 color='black'),
                       showlegend=False),
            row=1, col=coln + 1)

        # mean marker
        _ = fig.add_trace(go.Scatter(x=[x.mean()],
                                     y=[y.mean()],
                                     mode='markers',
                                     marker=dict(color='green',
                                                 size=4,
                                                 symbol='cross'),
                                     showlegend=False),
                          row=1, col=coln + 1)

        fig.add_vline(x=0, line=dict(color='darkgray', dash='dot', width=1),
                      opacity=1, row=1, col=coln + 1)
        fig.add_hline(y=0, line=dict(color='darkgray', dash='dot', width=1),
                      opacity=1, row=1, col=coln + 1)

    ####### Formating #######
    w, h = 4, 1  # inches
    fig.update_layout(template='simple_white',
                      width=96 * w, height=96 * h,
                      margin=dict(l=10, r=10, t=10, b=10))

    fig.update_xaxes(tickfont_size=9,
                     range=[-1, 1],
                     tickmode='array',
                     tickvals=[-1, 0, 1],
                     ticktext=[-1, 0, 1])

    fig.update_yaxes(tickfont_size=9,
                     range=[-1, 1],
                     tickmode='array',
                     tickvals=[-1, 0, 1],
                     ticktext=[-1, 0, 1])

    fig.update_xaxes(title=dict(text='firing rate MI',
                                font_size=10, standoff=0),
                     col=1, row=1)

    fig.update_yaxes(title=dict(text='contextual effects MI',
                                font_size=10, standoff=0),
                     col=1, row=1)

    return fig
