# ToDo document functions

import pathlib as pl
import joblib as jl

import pandas as pd
import numpy as np
import scipy.stats as sst
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from publication.globals import config, MINIMAL_DF

from src.visualization.palette import DURCOLOR, AMPCOLOR
from src.utils.dataframes import kruskal_with_posthoc
from src.metrics.context_metric_correlation import (
    calculate_metric_correlation_null_distribution
)

#### generates time shuffles fo the metric correlation analysis  ####
# Runs the shuffles, and caches results, or loads from cache
cache_file = pl.Path(
    config['paths']['analysis_cache']) / '230921_metric_correlations'
recache = False
n_shuffles = 1000  # number of time shuffles for each context instance

if cache_file.exists() and not recache:
    # loads cache
    print(f'found cache at {cache_file}, loading...')
    (calc_r_value,
     null_r_distr,
     pvalue) = jl.load(cache_file)
    print('...done')

else:
    # calculates and caches
    print(f'Calculating context effect metric correlation and null '
          f'distribution')
    (calc_r_value,
     null_r_distr,
     pvalue) = calculate_metric_correlation_null_distribution(
        n_shuffles=1000
    )

    print(f"Cacheing to {cache_file}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    jl.dump(
        (calc_r_value, null_r_distr, pvalue),
        cache_file
    )
    print('...done')

#### Formats layer and context effects together ####
# loads cellid layer information and merges to minimal context effect dataframe

csvpath = pl.Path(
    config['paths']['analysis_cache']
) / '231026_depth_info.csv'

depth_df = pd.read_csv(csvpath)
depth_df.rename(columns={"Unnamed: 0": 'id'}, inplace=True)

layer_ctx_effect_DF = pd.merge(
    left=depth_df.loc[:, ('id', 'layer')],
    right=MINIMAL_DF,
    on="id",
    validate="1:m"
).replace(
    {'layer': {56: '5-6',
               4: '4',
               13: '1-3'}},
).query(
    "value > 0 and analysis == 'SC'"
).dropna()


def plot_context_metric_correlation_significance():
    """
    Supplementary figure x. Shows the null distribution of the correlation
    between the context effect metrics Amplitude and duration, as a histogram
    in black, and the real correlation measured between these metrics in green.
    The null distribution was obtained by shuffling time bins for all
    significant context instances and recalculating the context effects
    metrics.
    While the null distribution shows a  significant positive correlation,
    confirming that our metrics are correlated by design, the measured
    correlation is significantly greater, showing a real physiological
    correlation between the metrics.

    Returns: Plotly Figure.

    """

    # prints the mean of the null distribution and of the calculated value
    # for record keeping and parsing into paper.
    print("##### R null distribution mean and SEM #####\n",
          np.mean(null_r_distr), sst.sem(null_r_distr))
    print("##### Measured R value #####\n", calc_r_value)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=null_r_distr, histfunc='count', histnorm='percent',
            marker=dict(color='black', line_width=0),
            name='null distribution'
        )
    )

    #
    fig.add_trace(
        go.Scatter(
            x=[calc_r_value, calc_r_value], y=[0, 8],
            mode='lines',
            line=dict(color='green', width=3),
            name=f'measured correlation<br>p-value={pvalue}'
        )
    )

    fig.update_layout(
        template='simple_white',
        width=96 * 3, height=96 * 1.5,
        margin=dict(t=10, b=10, l=10, r=10),
        bargap=0,
        xaxis=dict(title=dict(standoff=0,
                              text="Pearson's R",
                              font_size=10),
                   tickfont_size=9,
                   range=[0.19, 0.51]),
        yaxis=dict(title=dict(standoff=0,
                              text='percentage',
                              font_size=10),
                   tickfont_size=9,
                   range=[0, 8.5]),
        showlegend=True,
        legend=dict(xanchor='right', x=1, yanchor='top', y=1,
                    font_size=8, title=dict(text=''))
    )

    return fig


def plot_layer_effect_on_context_metrics():
    """

    Returns:

    """
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,
                        shared_xaxes=False, shared_yaxes=False)

    print(f"{len(layer_ctx_effect_DF.id.unique())} neurons labeled by layer")
    print(
        layer_ctx_effect_DF.loc[:, ('id', 'layer')].drop_duplicates().groupby(
            'layer'
        ).agg(count=('id', 'count'))
    )

    for rr, met in enumerate(['integral', 'last_bin']):
        print(f"\n#### {met} ####")
        func_df = layer_ctx_effect_DF.query(f"metric == '{met}'")

        _ = kruskal_with_posthoc(func_df, group_col='layer',
                                 val_col='value')
        func_df = func_df.groupby('layer').agg(count=('value', 'count'),
                                               stat=('value', np.mean),
                                               err=('value', sst.sem))
        print('summary stats\n', func_df, '\n')

        x = ['1-3', '4', '5-6']
        y = [func_df.at[cat, 'stat'] for cat in x]
        yerr = [func_df.at[cat, 'err'] for cat in x]
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode='markers',
                marker=dict(color='black', size=4),
                error_y=dict(array=yerr, color='black',
                             thickness=1, width=5),
                showlegend=False
            ),
            col=rr + 1, row=1
        )

    h, w = 1.3, 2.5
    fig.update_layout(
        template='simple_white', width=96 * w, height=96 * h,
        margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        font_family='Arial',

        xaxis=dict(title=dict(text="cortical layer")),

        yaxis=dict(matches=None, autorange=True,
                   title=dict(text='amplitude<br>(Delta Z-score*s)',
                              font_color=AMPCOLOR)),
        yaxis2=dict(matches=None, autorange=True,
                    title=dict(text='duration (ms)',
                               font_color=DURCOLOR)),
    )

    fig.update_xaxes(title_font_size=10, title_standoff=0, tickfont_size=9)
    fig.update_yaxes(title_font_size=10, title_standoff=0, tickfont_size=9)

    return fig
