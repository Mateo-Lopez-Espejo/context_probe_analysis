# todo change file name once order in paper is defined

import pathlib as pl
import plotly.graph_objects as go
import joblib as jl

from publication.globals import config
from src.metrics.context_metric_correlation import (
    calculate_metric_correlation_null_distribution
)

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
