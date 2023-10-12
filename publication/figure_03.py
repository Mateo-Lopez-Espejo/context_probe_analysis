"""
Figure 3 all panels
"""
import itertools as itt

import pandas as pd
import numpy as np
from scipy.stats import sem, wilcoxon, mannwhitneyu
import plotly.graph_objects as go

from src.visualization.palette import A1_COLOR, PEG_COLOR
from src.visualization.interactive import plot_site_coverages
from publication.globals import SC_DF, PCA_DF, DF_f3 as DF, ferret_df

def plot_example_site_coverage():
    """
    Figure 3 panels A and B. Coverage plots showing all context instances
    (a context pair and a probe) on a 2d plane, with a color map showing the
    context effects amplitude for each instance. The same heatmap is plot for
    multiple example neurons in a site, and for the first PC and Union
    heuristics of site coverage.

    Returns: Plotly figure

    """
    eg_site = 'ARM021b' # site from the example cells in figure 2
    eg_neurons = ['06-3', '10-4', '36-8', '40-6', '43-8', '53-1']
    eg_neurons = [f"{eg_site}-{cid}" for cid in eg_neurons]

    # print sound names associated with numbers
    print("names of sounds used:\n",
        ferret_df.query(
            f"id == '{eg_neurons[0]}'"
        ).loc[
                    :, ('probe', 'named_probe')
                    ].drop_duplicates().reset_index(drop=True)
    )


    eg_site_df = SC_DF.query(f"site == '{eg_site}'"
                             f" and metric == 'integral'"
                             f" and analysis == 'SC'")

    # selects subset of example neurons
    toplot_df = eg_site_df.query(
        f"id in {eg_neurons}"
    ).loc[:, ["id", "context_pair", "probe", "value"]].copy()

    print('Coverage counts:\n',
          toplot_df.groupby('id').agg(count=('value', lambda x: np.sum(x > 0))
                                      ).sort_values(by='count', ascending=False
                                                    ))

    # first principal component
    toplot_pca = PCA_DF.query(f"site == '{eg_site}' and PC == 1").copy()

    # union
    toplot_union = DF.query(
        f"analysis == 'SC' and mult_comp_corr == 'bf_ncp'"
        f" and site == '{eg_site}' and metric == 'integral'"
        f" and value > 0"
    ).groupby(
        by=["context_pair", "probe"], observed=True
    ).agg(
        value=('value', 'max'), text=('value', 'count')
    ).reset_index()

    # add dummy IDs to be used as subtitles
    toplot_union['id'] = 'Union'
    toplot_pca['id'] = 'PC1'

    # hack, concatenate Union and PCA as if single units,
    # to be handled similarly by the plotting function
    toplot_df = pd.concat([toplot_df, toplot_pca, toplot_union])
    fig = plot_site_coverages(toplot_df, rows=1, cols=8)

    fig.update_layout(height=96*2, width=96*6)

    return fig

def plot_neuron_and_site_coverage_summary():
    """
    Figure 3 panel C. Violin plots summarizing the context coverage for single
    neurons, the best neuron in the site, the first PC and the Union.
    Further categorization is done by different cortical regions. Additionally
    prints the nonparametric statistical tests comparing between cortical
    regions for each category, and between categories.
    For the category comparisons, either a matched sample statistic is used
    between site coverages, or an unmatched sample stat is used when comparing
    sites with single neurons.

    Returns: Plotly figure

    """
    def union_cover(df):
        df = df.pivot_table(index='id', columns=['context_pair', 'probe'],
                            values='value', aggfunc='first', observed=True)
        arr =  df.dropna(axis=1).values > 0
        union = np.any(arr, axis=0)
        return union.sum()/union.size * 100

    def best_neuron(df):
        df = df.pivot_table(index='id', columns=['context_pair', 'probe'],
                            values='value', aggfunc='first', observed=True)
        arr =  df.dropna(axis=1).values > 0
        neu_cvr = arr.sum(axis=1) / arr.shape[1] * 100
        return neu_cvr.max()

    # all single neuron coverages
    by_neuron = DF.query(
        "analysis == 'SC' and mult_comp_corr == 'bf_cp' "
        "and metric == 'integral'"
    ).groupby(
        ['region', 'site', 'id'], observed=True
    ).agg(
        value=('value', lambda x: np.sum(x>0)/x.size *100)
    )

    # best neuron in site
    by_best =  DF.query(
        "analysis == 'SC' and mult_comp_corr == 'bf_cp' "
        "and metric == 'integral'"
    ).groupby(['region', 'site'], observed=True).apply(best_neuron)
    by_best.name = 'value'

    # Union of single neurons, note the more strict bonferroni
    by_union =  DF.query(
        "analysis == 'SC' and mult_comp_corr == 'bf_ncp' "
        "and metric == 'integral'"
    ).groupby(['region', 'site'], observed=True).apply(union_cover)
    by_union.name = 'value'

    # First PC coverage
    by_PC1 = DF.query(
        "analysis == 'PCA' and PC == 1 and "
        "mult_comp_corr == 'bf_cp' and metric == 'integral'"
    ).groupby(
        ['region', 'site'], observed=True
    ).agg(
        value=('value', lambda x: np.sum(x>0)/x.size *100)
    )

    # concatenate dissimilar data in single DF for quicker plotly express
    to_concat = {
        'SC':by_neuron, 'best_SC':by_best ,'union': by_union, 'PC1':by_PC1
    }
    toplot = list()
    for name, df in to_concat.items():
        df = df.reset_index()
        df['quant'] = name
        toplot.append(df)
    toplot = pd.concat(toplot)

    ################# Print relevant statistical comparisons #################
    print(
    "\n######################################################################"
    "\n############# Comparison between coverage category       #############"
    "\n############# adjusting for matched sample number or not #############"
    )
    # compare between quantifications
    stats_df = list()
    for q0, q1 in itt.combinations(toplot.quant.unique(), 2):
        if 'SC' in [q0, q1]:
            x = toplot.query(f"quant == '{q0}'").value.values
            y = toplot.query(f"quant == '{q1}'").value.values
            assert np.all(~np.isnan(x))
            assert np.all(~np.isnan(y))
            out = mannwhitneyu(x, y)
            stat = 'mannwhitneyu'
        else:
            ddd = toplot.query(
                f"quant in {[q0, q1]}"
            ).pivot(
                index='site', columns='quant', values='value'
            )
            x = ddd[q0].values
            y = ddd[q1].values
            assert np.all(~np.isnan(x))
            assert np.all(~np.isnan(y))
            out = wilcoxon(x, y)
            stat = 'wilcoxon'

        d = {'pair': f'{q0}_vs_{q1}', 'stat': stat, 'pvalue': out.pvalue,
             'x_mean': np.mean(x), 'x_sem': sem(x), 'y_mean': np.mean(y),
             'y_sem': sem(y)}
        stats_df.append(d)

    stats_df = pd.DataFrame(stats_df)
    alpha = 0.01 / stats_df.shape[0]
    stats_df['signif'] = stats_df['pvalue'] < alpha
    print(stats_df)

    print(
    "\n######################################################################"
    "\n################ Comparison between cortical regions #################"
    "\n################ within each coverage category       #################"
    )
    stats_df = list()
    for quant in toplot.quant.unique():
        if quant == 'SC':
            idx = 'id'
        else:
            idx = 'site'

        x = toplot.query(f"quant == '{quant}' and region == 'A1'").value.values
        y = toplot.query(
            f"quant == '{quant}' and region == 'PEG'").value.values

        assert np.all(~np.isnan(x))
        assert np.all(~np.isnan(y))
        out = mannwhitneyu(x, y)
        stat = 'mannwhitneyu'

        d = {'quant': quant, 'stat': stat, 'pvalue': out.pvalue,
             'A1mean': np.mean(x), 'A1SEM': sem(x),
             'PEGmean': np.mean(y), 'PEGSEM': sem(y)}
        stats_df.append(d)

    stats_df = pd.DataFrame(stats_df)
    alpha = 0.01 / stats_df.shape[0]
    stats_df['signif'] = stats_df['pvalue'] < alpha
    print(stats_df)

    ##################### Now starts building the figure #####################
    fig = go.Figure()

    # violins plust pointplots
    for qq, quant in enumerate(['SC', 'PC1', 'best_SC', 'union']):
        fig.add_trace(
            go.Violin(
                x=toplot['quant'][(toplot['region'] == 'A1') &
                                  (toplot['quant'] == quant)],
                y=toplot['value'][(toplot['region'] == 'A1') &
                                  (toplot['quant'] == quant)],
                legendgroup='A1', scalegroup=f'', name='A1',
                side='negative',
                pointpos=-0.3,  # where to position points
                line=dict(color=A1_COLOR,
                          width=1),
                showlegend=False,
                meanline=dict(width=1, )
            )
        )
        fig.add_trace(
            go.Violin(
                x=toplot['quant'][(toplot['region'] == 'PEG') &
                                  (toplot['quant'] == quant)],
                y=toplot['value'][(toplot['region'] == 'PEG') &
                                  (toplot['quant'] == quant)],
                legendgroup='PEG', scalegroup=f'', name='PEG',
                side='positive',
                pointpos=0.3,
                line=dict(color=PEG_COLOR,
                          width=1),
                showlegend=False,
                meanline=dict(width=1, )
            )
        )

    # update characteristics shared by all traces
    fig.update_traces(meanline_visible=True,
                      marker_size=2,
                      points='all',
                      jitter=0.1,
                      scalemode='width',
                      spanmode='hard')

    # add lines connecting dots for groups
    linearr = toplot.query(
        "quant in ['best_SC', 'PC1', 'union']"
    ).pivot(index='site', columns='quant', values='value'
            ).loc[:, ['PC1', 'best_SC', 'union']].values

    # individual grayed out lines
    x = ['PC1', 'best_SC', 'union']
    for ll, line in enumerate(linearr):
        fig.add_trace(go.Scatter(x=x, y=line,
                                 mode='lines',
                                 opacity=0.1,
                                 line=dict(
                                     color='gray',
                                     width=1)
                                 ,
                                 showlegend=False)
                      )

    # mean values plus error bars
    mean = linearr.mean(axis=0)
    print(f'best_SC={mean[0]:.2f} PC1={mean[1]:.2f} union={mean[2]:.2f}')
    err = sem(linearr, axis=0)
    fig.add_trace(go.Scatter(x=x, y=mean,
                             mode='markers+lines',
                             opacity=1,
                             showlegend=False,
                             marker=dict(color='black',
                                         symbol='square',
                                         size=2,
                                         line=dict(color='black',
                                                   width=1)
                                         ),
                             error_y=dict(array=err,
                                          color='black',
                                          thickness=1,
                                          width=5
                                          ),
                             line=dict(color='black',
                                       width=1
                                       )
                             ))

    # mean for the single cell pooled values
    SC_arr = toplot.query("quant == 'SC'").value.values
    mean = SC_arr.mean()
    print(f'SC={mean}')
    err = sem(SC_arr)

    fig.add_trace(go.Scatter(x=['SC'], y=[mean],
                             mode='markers',
                             opacity=1,
                             showlegend=False,
                             marker=dict(color='black',
                                         symbol='square',
                                         size=2,
                                         line=dict(color='black',
                                                   width=1)
                                         ),
                             error_y=dict(array=[err],
                                          color='black',
                                          thickness=1,
                                          width=5
                                          ),
                             ))

    # figure size and labels formating

    w, h = 3 * 96, 2.5 * 96
    fig.update_layout(template="simple_white",
                      width=w, height=h,
        margin=dict(l=40, r=10, t=10, b=20),
        xaxis=dict(range=[-0.6, 3.6],
                   title=dict(text='', standoff=0)),
        yaxis=dict(range=[0, 100],
                   title=dict(text='coverage %', standoff=0)),
        violingap=0, violingroupgap=0, violinmode='overlay',
        )

    return fig
