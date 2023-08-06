"""
Figure 2 all panels, information printing function,
and supplementary figure 1
"""
import pandas as pd
import joblib as jl
import numpy as np
import scipy.stats as sst
from statsmodels.formula.api import ols
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.colors as pc

from publication.globals import summary_DF_file, pivoted, toplot, toregress, \
    ferret_df

from src.utils.dataframes import kruskal_with_posthoc
from src.utils.tools import decimate_xy
from src.visualization.palette import REGION_COLORMAP, DURCOLOR, AMPCOLOR



def print_percentage_significant():
    """
    Information regarding total number of instances, significant percentage and
    neurons that show any significant effects.
    Prints:

    45906/501440, 9.155% significant  instances
    1232/2728, 45.161% modulated neurons

    And then groups by region and context
    transition pair types.

    Returns: None

    """
    DF = jl.load(summary_DF_file).query(
        "analysis == 'SC' and mult_comp_corr == 'bf_cp' and metric in 'integral' ")
    # number proportion of significant instances
    nsignif = np.sum(DF.value.values > 0)
    ntotal = DF.shape[0]
    print(
        f'{nsignif}/{ntotal}, {nsignif / ntotal * 100:.3f}% significant  instances')

    # proportion of neurons with at least one signif instance
    neusig = DF.groupby('id').agg(modulated=('value', lambda x: np.any(x > 0)))
    nsignif = np.sum(neusig.modulated.values)
    ntotal = neusig.shape[0]
    print(
        f'{nsignif}/{ntotal}, {nsignif / ntotal * 100:.3f}% modulated neurons')

    # count of singificant instances grouping by category levels
    inst_by_reg = pivoted.groupby('region', observed=True).agg(
        inst_count=('integral', 'count'), neu_count=('id', pd.Series.nunique))
    print(inst_by_reg)

    inst_by_trans = pivoted.groupby('trans_pair', observed=True).agg(
        inst_count=('integral', 'count'), neu_count=('id', pd.Series.nunique))
    print('\n', inst_by_trans)

    return None


def plot_duration_vs_amplitude_scatter():
    """
    Figure 2 panel A. Scatter of 2000 random example instances showing their
    duration and amplitude. Colored by region (1000, each). With the two
    example instances from figure 1 highligted in black.
    The function Also prints the linear regression metrics between the
    measures, the number of instances with context effects at the last bin and
    general statistics of Amplitude and duration for the full dataset.

    Returns: Plotly figure

    """
    ### x, y regression and number of x saturated (x==1000) points
    print(f"######## full dataset metrics ######## "
          f"\n{sst.linregress(pivoted.last_bin, pivoted.integral)}"
          f"\ntime saturated instances: n={pivoted.query('last_bin == 1000').shape[0]}, "
          f"%={pivoted.query('last_bin == 1000').shape[0] / pivoted.shape[0] * 100}"
          f"\n{pivoted.loc[:, ['integral', 'last_bin']].describe()}")

    fig = go.Figure()
    maxy = list()
    rng = np.random.default_rng(42)
    for region in ['A1', 'PEG']:
        toplot = pivoted.query(f"region == '{region}'")
        x = toplot['last_bin_jittered'].values
        y = toplot['integral'].values

        xm, ym = decimate_xy(x, y, end_num=1000, by_quantiles=False, rng=rng)
        maxy.append(np.max(ym))

        fig.add_trace(go.Scatter(x=xm, y=ym, mode='markers', opacity=0.5,
                                 marker=dict(color=REGION_COLORMAP[region],
                                             opacity=0.9, size=2), name=region,
                                 hoverinfo='skip', showlegend=False))

    maxy = np.max(maxy)

    ## highlight the special cell, defined in the first cell of the notebook
    # same examples as in figure 1, cellid, (context-pair,), probe
    example_cells = [('ARM021b-36-8', (0, 1), 3),
                     ('ARM021b-36-8', (1, 3), 4), ]

    for cellid, ctx_pair, probe in example_cells:
        celltoplot = toplot.query(
            f"id == '{cellid}' and context_pair == '{ctx_pair[0]:02}_{ctx_pair[1]:02}' and "
            f"probe == {probe}")

        fig.add_trace(go.Scatter(x=celltoplot['last_bin_jittered'],
                                 y=celltoplot["integral"], mode='markers',
                                 marker=dict(color=REGION_COLORMAP[
                                     celltoplot['region'].values[0]], size=4,
                                             line=dict(color='black',
                                                       width=1)),
                                 hoverinfo='skip', showlegend=False))

    w, h = 2.5, 1.5  # in inches
    _ = fig.update_layout(template='simple_white',
                          margin=dict(l=10, r=10, t=10, b=10),
                          width=round(96 * w), height=round(96 * h),

                          xaxis=dict(range=[0, 1050],
                                     title=dict(text='Duration (ms)',
                                                standoff=0,
                                                font_color=DURCOLOR,
                                                font_size=10),
                                     tickfont_size=9),

                          yaxis=dict(range=[0, maxy + maxy * 0.01], title=dict(
                              text="Amplitude (Delta Z-score*s) ", standoff=0,
                              font_color=AMPCOLOR, font_size=10),
                                     tickfont_size=9))

    return fig


def print_metrics_by_classification():
    """
    Prints mean and SEM for the Amplitude and Duration of the different levels
    of the 3 clasification by: Region, transition-pair and vocalization type.

    Returns: None

    """
    #### metrics by region formated for paper
    print("\n##### mean and error values by region #####")
    toprint = toregress.groupby(['metric', 'region'], observed=True).agg(
        mean=('value', np.mean), std=('value', np.std), sem=('value', sst.sem))
    for metric in toprint.index.levels[0]:
        inner = toprint.loc[metric, :]
        str_parts = list()
        for tp, vals in inner.iterrows():
            str_parts.append(
                f"{tp}: {vals.loc['mean']:.2f}±{vals.loc['sem']:.3f}")

        if metric == 'integral':
            units = 'Z-score*s'
        elif metric == 'last_bin':
            units = 'ms'

        full_str = f'{metric} mean±SEM. ' + ', '.join(str_parts) + units
        print(full_str)

    #### metrics by vocalization formated for paper
    print("\n##### mean and error values by vocalization #####")
    # filters out silence instances to account for interference
    # interaction with vocalization
    toprint = toregress.query('silence == 0').groupby(
        ['metric', 'voc_ctx']).agg(mean=('value', np.mean),
                                   std=('value', np.std),
                                   sem=('value', sst.sem))
    for metric in toprint.index.levels[0]:
        inner = toprint.loc[metric, :]
        str_parts = list()
        for tp, vals in inner.iterrows():
            str_parts.append(
                f"{tp}: {vals.loc['mean']:.2f}±{vals.loc['sem']:.4f}")

        if metric == 'integral':
            units = 'Z-score*s'
        elif metric == 'last_bin':
            units = 'ms'

        full_str = f"{metric} mean±SEM. {', '.join(str_parts)} {units}"
        print(full_str)

    #### metrics by context pair formated for paper
    print("\n##### mean and error values by transition #####")
    # filters out the silence vocalization context pairs
    toprint = toplot.query(
        "not (transition == 'silence' and voc_ctx == 'vocalization')").groupby(
        ['metric', 'transition']).agg(mean=('value', np.mean),
                                      std=('value', np.std),
                                      sem=('value', sst.sem))
    for metric in toprint.index.levels[0]:
        inner = toprint.loc[metric, :]
        str_parts = list()
        for tp, vals in inner.iterrows():
            str_parts.append(
                f"{tp}: {vals.loc['mean']:.2f}±{vals.loc['sem']:.4f}")
        if metric == 'integral':
            units = 'Z-score*s'
        elif metric == 'last_bin':
            units = 'ms'
        full_str = f"{metric} mean±SEM. {', '.join(str_parts)} {units}"
        print(full_str)


def plot_all_context_pairs_grid():
    """
    Figure 2 panel B. Grid showing all combinations of possible contexts with
    their associated mean Amplitude and Duration, number e of instances and
    fraction of total instancese anotated. The 'null' text values and some of
    the least frequent instances were eliminated post hoc for clarity.
    Returns:

    """

    # organized the data to be plotted in square arrays for the amplitude
    # duration, total cound and fraction.

    ctx_class = ['silence', 'same-nonvoc', 'diff-nonvoc', 'same-voc',
                 'diff-voc']
    n_instances = ferret_df.query("metric == 'integral'").shape[0]
    meandf = ferret_df.groupby(['metric', 'class_pair']).agg(
        mean=('value', 'mean'), count=('value', 'count'),
        fract=('value', lambda x: x.shape[0] / n_instances * 100))

    amp_arr = np.full([len(ctx_class), len(ctx_class)], fill_value=np.nan)
    dur_arr = np.full([len(ctx_class), len(ctx_class)], fill_value=np.nan)

    count_arr = np.full([len(ctx_class), len(ctx_class)], fill_value=np.nan)
    frac_arr = np.full([len(ctx_class), len(ctx_class)], fill_value=np.nan)

    for i, ctx0 in enumerate(ctx_class):
        for j, ctx1 in enumerate(ctx_class):
            composite = [ctx0, ctx1]
            composite.sort()
            composite = '_'.join(composite)
            indf = meandf.query(f"class_pair == '{composite}'")
            if indf.empty:
                continue
            if i >= j:
                amp_arr[i, j] = indf.loc['integral', 'mean'].values
                frac_arr[i, j] = indf.loc['last_bin', 'fract'].values
            if i <= j:
                dur_arr[i, j] = indf.loc['last_bin', 'mean'].values
                count_arr[i, j] = indf.loc['integral', 'count'].values

    fig = make_subplots(1, 2, shared_xaxes='all', shared_yaxes='all',
                        horizontal_spacing=0.05, vertical_spacing=0.05)

    for cc, (Z, txt) in enumerate(
            zip([amp_arr, dur_arr], [frac_arr, count_arr])):
        if cc == 0:
            coloraxis = 'coloraxis'
            texttemplate = '%{text:.2f}'
        else:
            coloraxis = f'coloraxis{cc + 1}'
            texttemplate = '%{text}'
        f = go.Figure()
        f.add_trace(go.Heatmap(z=Z, x=ctx_class, y=ctx_class, xgap=1, ygap=1,
                               coloraxis=coloraxis, ))

        # here text has to be passed as a DF with context and probes as
        # indices and columns
        text_df = pd.DataFrame(data=txt, index=ctx_class, columns=ctx_class)

        f.update_traces(text=text_df, texttemplate=texttemplate, )

        fig.add_traces(f.data, rows=[1] * len(f.data),
                       cols=[1] * len(f.data))  # overlay in a sinlge pannel

    # darker verions of the colors above
    amp_cscale = pc.make_colorscale(['#FFFFFF', '#3B6B2F'])
    dur_cscale = pc.make_colorscale(['#FFFFFF', '#683F5D'])

    fig.update_layout(template='simple_white',
                      margin=dict(t=10, b=10, l=10, r=10), width=6 * 96,
                      height=3 * 96,
                      coloraxis=dict(showscale=True, colorscale=amp_cscale,
                                     colorbar=dict(orientation='v',
                                                   thicknessmode='fraction',
                                                   thickness=0.05,
                                                   lenmode='fraction', len=0.5,
                                                   title=dict(
                                                       text='Amplitude<br>'
                                                            '(Delta Z-Score)',
                                                       side='right',
                                                       font_size=10),
                                                   ticklen=5, tickangle=-50,
                                                   tickfont_size=9,
                                                   xanchor='left', x=1,
                                                   yanchor='top', y=1)),
                      coloraxis2=dict(showscale=True, colorscale=dur_cscale,
                                      colorbar=dict(orientation='v',
                                                    thicknessmode='fraction',
                                                    thickness=0.05,
                                                    lenmode='fraction',
                                                    len=0.5, title=dict(
                                              text='Duration (ms)',
                                              side='right', font_size=10),
                                                    ticklen=5, tickangle=-50,
                                                    tickfont_size=9,
                                                    xanchor='left', x=1,
                                                    yanchor='top', y=0.5), ))

    fig.update_yaxes(scaleanchor='x', constrain='domain')
    fig.update_yaxes(title_text='context 2', title_standoff=0, col=1, row=1)
    fig.update_xaxes(constrain='domain', title_text='context 1',
                     title_standoff=0)

    return fig


def plot_metric_by_categories_comparisons():
    """
    Figure 2 panels C to H. Point and wiskers plots showing meand and SEM for
    the Amplitude and Duration discriminated by cortical region, context
    similarity (silence, different, same) and context identity (vocalization
    and non-vocalization)

    Returns: Plotly figure

    """

    def _plot_metric_quant_bars(df, metric, category):
        """
        Base function to handle the plotting of individual subpanels corresponding
        to One metric ("integral" or "last_bin") and one category
        ("region", "transition", "vocalization"). The function returns a list of
        plotly traces ready to be added to a plotly figure.
        Notice the filtering of silence and silence+vocalization instances.
        This has been stated in the document and the interaction it eliminates
        its described by the multivariate linear regression performed below.

        Args:
            df: Pandas dataframe
            metric: str. 'integral' or 'last_bin'
            category: str. 'region', 'transition', 'vocalization'

        Returns: list of plotly traces

        """
        print(f'\n######### {metric}, {category} #########\n')
        df = df.query(f"metric == '{metric}'")

        if category == 'transition':
            ### transitions ###

            # filters out the silence vocalization context pairs
            df = df.query(
                "not (transition == 'silence' and voc_ctx == 'vocalization')")
            _ = kruskal_with_posthoc(df, group_col='transition',
                                     val_col='value')
            df = df.groupby(by=['transition'], observed=True).agg(
                stat=('value', np.mean), err=('value', sst.sem))

            x = ['same', 'diff', 'silence']
            y = [df.at[cat, 'stat'] for cat in x]
            yerr = [df.at[cat, 'err'] for cat in x]
            return [go.Scatter(x=x, y=y, mode='markers',
                               marker=dict(color='black', size=4),
                               error_y=dict(array=yerr, color='black',
                                            thickness=1, width=5),
                               showlegend=False)]

        ### regions ###
        elif category == 'region':
            _ = kruskal_with_posthoc(df, group_col='region', val_col='value')
            df = df.groupby('region').agg(stat=('value', np.mean),
                                          err=('value', sst.sem))

            # if you want different color error bars, have to do it one at a time
            reg_quant = list()
            for rr, row in df.iterrows():
                reg_quant.append(
                    go.Scatter(x=(rr,), y=(row.stat,), mode='markers',
                               marker=dict(color=REGION_COLORMAP[rr], size=4),
                               error_y=dict(array=(row.err,),
                                            color=REGION_COLORMAP[rr],
                                            thickness=1, width=5),
                               showlegend=False), )

            return reg_quant

        elif category == 'vocalization':
            ### vocalizations ###

            # filtering out silence such that it does not occlude vocalization
            # effects
            df = df.query('silence == 0')
            _ = kruskal_with_posthoc(df, group_col='voc_ctx', val_col='value')
            df = df.groupby(by=['voc_ctx'], observed=True).agg(
                stat=('value', np.mean), err=('value', sst.sem))

            x = ['sound', 'vocalization']
            y = [df.at[cat, 'stat'] for cat in x]
            yerr = [df.at[cat, 'err'] for cat in x]
            return [go.Scatter(x=x, y=y, mode='markers',
                               marker=dict(color='black', size=4),
                               error_y=dict(array=yerr, color='black',
                                            thickness=1, width=5),
                               showlegend=False)]


        else:
            raise ValueError(f'bad param {category}')

    fig = make_subplots(2, 3, column_width=[0.25, 0.5, 0.25],
                        shared_xaxes='columns', shared_yaxes='rows')
    for mm, met in enumerate(['integral', 'last_bin']):
        """
        here consider that the dataframe with duplicated data is only required 
        for transitions, and if used on the vocalizations, it inverts the 
        results since its probably multiplying silences more than voc.
        Notice the [toregress, toplot, toregress] below. 
        """
        for cc, (cat, fdf) in enumerate(
                zip(['region', 'transition', 'vocalization'],
                    [toregress, toplot, toregress])):
            pan = _plot_metric_quant_bars(fdf, met, cat)
            fig.add_traces(pan, cols=[cc + 1] * len(pan),
                           rows=[mm + 1] * len(pan))

    w, h = 2, 2 * 4.6 / 3.4  # plaing with aspect ratio
    fig.update_layout(template='simple_white', width=96 * w, height=96 * h,
                      margin=dict(l=10, r=10, t=10, b=10), showlegend=False, )

    fig.update_xaxes(title=dict(text='region'), col=1, row=2)
    fig.update_xaxes(title=dict(text='context type'), tickangle=-45, col=2,
                     row=2)
    fig.update_xaxes(title=dict(text='ethology'), tickangle=-45, col=3, row=2)

    fig.update_yaxes(title=dict(text="Amplitude", font_color=AMPCOLOR, ),
                     range=[0.2, 0.28],
                     # tickmode='linear', tick0=0.16, dtick=0.04,
                     col=1, row=1)
    fig.update_yaxes(title=dict(text='Duration', font_color=DURCOLOR),
                     range=[220, 290], col=1, row=2)

    fig.update_xaxes(title_font_size=10, title_standoff=0, tickfont_size=9)
    fig.update_yaxes(title_font_size=10, title_standoff=0, tickfont_size=9)

    return fig


def print_multivariate_linear_regressions():
    """
    Given the non-orthogonal nature of the clasification by context similarity
    and context type, a multivariate linear model, which can consider their
    interaction was formulated. For good form and completeness a similar model
    is done for the cortical region,

    Returns: None

    """

    print(
        '\n############## Multivariate linear regression models ##############'
        '\n####### by region #######\n')
    for metric in toregress.metric.unique().tolist():
        print(f"\n#### {metric} ####")
        mod = ols("norm_val ~ C(region)",
                  data=toregress.query(f"metric == '{metric}'"))
        res = mod.fit()
        print(res.summary())

    # here merging all transitions in a single column, thus duplicating
    # data, but leaving vocalization as an orthogonal variable
    toreg = list()  # concatenate data classified on single values
    for cat in ['diff', 'same', 'silence']:
        subset = toregress.query(f"{cat} == 1").copy()
        subset['transition'] = cat
        toreg.append(subset)

    toreg = pd.concat(toreg)

    print('\n####### by context type and vocalization #######\n')
    for metric in toreg.metric.unique().tolist():
        print(f"\n#### {metric} ####")
        mod = ols("norm_val ~ C(voc_ctx) * C(transition)",
                  data=toreg.query(f"metric == '{metric}'"))
        res = mod.fit()
        print(res.summary())

    return None


def plot_metric_by_probe_vocalization():
    """
    Supplementary Figure 1 panels A and B. Point and wiskers plots showing
    meand and SEM for the Amplitude and Duration discriminated by
    whether the probe was a vocalization or not.

    Returns: Plotly figure.

    """

    # print mean and SEM to report on paper
    print("\n##### mean and error values by vocalization #####")
    toprint = toregress.groupby(['metric', 'voc_prb']).agg(
        mean=('value', np.mean), std=('value', np.std), sem=('value', sst.sem))
    for metric in toprint.index.levels[0]:
        inner = toprint.loc[metric, :]
        str_parts = list()
        for tp, vals in inner.iterrows():
            str_parts.append(
                f"{tp}: {vals.loc['mean']:.2f}±{vals.loc['sem']:.4f}")
        if metric == 'integral':
            units = 'Z-score*s'
        elif metric == 'last_bin':
            units = 'ms'
        full_str = f"{metric} mean±SEM. {', '.join(str_parts)} {units}"
        print(full_str)

    fig = make_subplots(2, 1, shared_xaxes='columns', shared_yaxes='rows')

    for mm, met in enumerate(['integral', 'last_bin']):
        print(f'\n######### {met} voc_prb #########\n')
        indf = toregress.query(f"metric == '{met}'")

        _ = kruskal_with_posthoc(indf, group_col='voc_prb', val_col='value')
        indf = indf.groupby(by='voc_prb', observed=True).agg(
            stat=('value', np.mean), err=('value', sst.sem),
            count=('value', 'count'))

        print('\ndata summary stats\n', indf)

        x = ['sound', 'vocalization']
        y = [indf.at[cat, 'stat'] for cat in x]
        yerr = [indf.at[cat, 'err'] for cat in x]

        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(color='black', size=4),
                                 error_y=dict(array=yerr, color='black',
                                              thickness=1, width=5),
                                 showlegend=False),
                      col=1, row=mm + 1)

    fig.update_layout(template='simple_white', height=96 * 2, width=96 * 1.5,

                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis2=dict(title=dict(text='probe')), yaxis=dict(
            title=dict(text="Amplitude", font_color=AMPCOLOR)), yaxis2=dict(
            title=dict(text='Duration', font_color=DURCOLOR)),

                      showlegend=False, )

    fig.update_xaxes(title_font_size=10, title_standoff=0, tickfont_size=9)
    fig.update_yaxes(title_font_size=10, title_standoff=0, tickfont_size=9)

    return fig


def print_probe_voc_regression():
    """

    Returns:

    """
    print('\n####### Linear regression by vocalization if in probe #######\n')
    for metric in toregress.metric.unique().tolist():
        print(f"\n#### {metric} ####")
        mod = ols("norm_val ~ C(voc_prb)",
                  data=toregress.query(f"metric == '{metric}' "))
        res = mod.fit()
        print(res.summary())

    return None
