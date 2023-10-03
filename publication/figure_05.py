import numpy as np
import scipy.stats as sst
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nems import db as nd
from nems_lbhb.baphy_experiment import BAPHYExperiment

from src.visualization.palette import (
    add_opacity, PHOTOACT, NAROWSPIKE, BROADSPIKE, AMPCOLOR, DURCOLOR
)

from src.utils.dataframes import kruskal_with_posthoc
from publication.globals import (
    toclust_f5 as toclust,  toregress_f5 as toregress, toplot_f5 as toplot,
    hist_threshold, margin,
)


def jknf(x, njacks=20, fn=np.mean):
    pred = x
    chunksize = int(np.ceil(len(pred) / njacks / 10))
    chunkcount = int(np.ceil(len(pred) / chunksize / njacks))
    idx = np.zeros((chunkcount, njacks, chunksize))
    for jj in range(njacks):
        idx[:, jj, :] = jj
    idx = np.reshape(idx, [-1])[:len(pred)]
    jc = np.zeros(njacks)
    for jj in range(njacks):
        ff = (idx != jj)
        jc[jj] = fn(pred[ff])

    stat = np.nanmean(jc)
    error = np.nanstd(jc) * np.sqrt(njacks - 1)

    return stat, error

def plot_cell_type_distributions():
    """
    Figure 5 panel C. Shows a histogram of number of neurons per spike width
    showcasing the clear bimodal distribution and coloring by spike width and
    optotagged neurons

    Returns: Plotly figure

    """

    print(f'waveform threshold. lower: {hist_threshold - margin:.2f}, '
          f'upper {hist_threshold + margin:.2f}')

    fig = px.histogram(
        toclust.dropna(subset=['triple']), nbins=100, x='sw',
        color='triple', pattern_shape='triple',
        pattern_shape_map={'activated': "", 'narrow': "",
                           "unclass": "x", "broad": ""},
        barmode='stack',
        color_discrete_map={'activated': PHOTOACT,
                            'narrow': NAROWSPIKE,
                            "unclass": 'lightgray',
                            "broad": BROADSPIKE},
        category_orders={'triple': ['activated', 'narrow', 'unclass', 'broad']}
    )
    fig.update_traces(marker_line_width=0)
    # This scaling is a hardcoded hack to bring the KDE to the border of the
    # histogram bars. It does not affect the KDE defined threshold and the
    # downstream analysis, but it does look cuter.
    kernel = sst.gaussian_kde(toclust['sw'], 0.1)
    x = np.linspace(0, 1.5, 100)
    y = kernel(x) * 32
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                             line=dict(color='dimgray', dash='dot'),
                             showlegend=False))
    fig.add_vline(x=hist_threshold,
                  line=dict(color='black', dash='dash', width=1), opacity=0.5)
    fig.add_vline(x=hist_threshold - margin,
                  line=dict(color='black', dash='dash', width=1), opacity=1)
    fig.add_vline(x=hist_threshold + margin,
                  line=dict(color='black', dash='dash', width=1), opacity=1)

    w, h = 3, 2
    fig.update_layout(
        template='simple_white', width=96 * w, height=96 * h,
        margin=dict(t=10, b=10, l=10, r=10), bargap=0,
        xaxis=dict(title=dict(standoff=0,
                              text='Peak-to-trough delay (ms)',
                              font_size=10),
                   tickfont_size=9),
        yaxis=dict(title=dict(standoff=0,
                              text='neuron count',
                              font_size=10),
                   tickfont_size=9),
        legend=dict(xanchor='right', x=1, yanchor='top', y=1,
                    font_size=9, title=dict(text=''))
    )
    return fig


def plot_classified_waveforms():
    """
    Figure 5 panel B. Displays a random subset of 150 waveforms for each neuron
    classification, coloring by said classification and computing the group
    average waveform

    Returns: Plotly figure

    """

    def get_aligned_waveforms(DF):
        wf = DF['waveform_norm'].values
        trough = DF['trough'].values

        centered = []
        fs = 10000 / (82 / 30000)
        for w, wave in enumerate(wf):
            wave = np.asarray(wave)
            t = int(trough[w])
            wave = wave[t - int(fs * 0.0005):t + int(fs * 0.001)]
            time = np.linspace(-.5, 1, wave.shape[0])
            centered.append(wave)

        centered = np.stack(centered, axis=0)

        return centered, time

    fig = go.Figure()
    for clss, color in zip(['broad', 'narrow', 'activated'],
                           [BROADSPIKE, NAROWSPIKE, PHOTOACT]):
        lines, t = get_aligned_waveforms(toclust.query(f"triple == '{clss}'"))

        # Single waveform examples, decimated,
        decimate = 150
        if lines.shape[0] >= decimate:
            decimator = np.random.choice(lines.shape[0], decimate,
                                         replace=False)
            mlines = lines[decimator, :]
        else:
            mlines = lines

        for line in mlines:
            _ = fig.add_trace(
                go.Scatter(x=t, y=line, mode='lines',
                           line=dict(color=add_opacity(color, 1),
                                     width=1),
                           opacity=0.5, showlegend=False)
            )

    # Average for the group, have to do it second so they are on top
    for clss, color in zip(['broad', 'narrow', 'activated'],
                           [BROADSPIKE, NAROWSPIKE, PHOTOACT]):
        lines, t = get_aligned_waveforms(toclust.query(f"triple == '{clss}'"))
        _ = fig.add_trace(
            go.Scatter(x=t, y=lines.mean(axis=0), mode='lines',
                       line=dict(color=color, width=3),
                       name=clss)
        )

    # formating
    w, h = 2, 2
    fig.update_layout(
        template='simple_white', width=96 * w, height=96 * h,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(title=dict(standoff=0, text='ms', font_size=9),
                   tickfont_size=9),
        yaxis=dict(title=dict(text='',
                              standoff=0),
                   showticklabels=False, showline=False,
                   ticks='', tickfont_size=9),
        showlegend=True,
        legend=dict(xanchor='right', x=1, yanchor='bottom', y=0,
                    font_size=9, title_text='',
                    bgcolor="rgba(0,0,0,0)")
    )

    return fig


def plot_optotagged_example_neuron():
    cellid = 'TNC013a-031-4'
    expt_n = 1  # which of the phototag experiments to use
    # data time offsets in ms? relative to light on
    tstart = -0.02
    tend = 0.1

    rasterfs = 5000

    # find the raw files in the database
    # to allow longer trials with short pulses, sub in
    # Trial_LightPulseDuration for Ref_Duration
    df = nd.pd_query(
        "SELECT sCellFile.cellid, sCellFile.stimfile, "
        "sCellFile.stimpath, sCellFile.rawid,"
        "g2.value as Ref_Duration "
        "FROM sCellFile "
        "INNER JOIN gData ON gData.rawid=sCellFile.rawid "
        "AND gData.name='TrialObjectClass' "
        "INNER JOIN gData g2 ON g2.rawid=sCellFile.rawid "
        "AND g2.name='Ref_Duration' "
        "WHERE gData.svalue='RefTarOpt' "
        "AND g2.value<0.1 "
        "AND sCellFile.RunClassid = 51 "
        f"AND sCellFile.cellid='{cellid}'"
    )

    # clean up DF
    df['siteid'] = df.cellid.apply(nd.get_siteid)
    df['recording'] = df.stimfile.str.split('.').str[0]
    df['parmfile'] = df.stimpath + df.stimfile  # full path to parameter file.
    df.drop(columns=['stimfile'], inplace=True)

    # select just one of multiple phototag experiments for the neuron
    paramfile, rawid = df.loc[:, ('parmfile', 'rawid')].iloc[expt_n, :]
    manager = BAPHYExperiment(parmfile=paramfile, rawid=rawid, cellid=cellid)

    rec = manager.get_recording(recache=True, resp=True, rasterfs=rasterfs,
                                stim=False)
    rec['resp'] = rec['resp'].rasterize()
    prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1] / rasterfs

    # get light on / off
    opt_data = rec['resp'].epoch_to_signal('LIGHTON')
    opto_mask = opt_data.extract_epoch('REFERENCE').any(axis=(1, 2))

    opt_start_stop_bins = np.argwhere(np.diff(
        opt_data.extract_epoch('REFERENCE')[opto_mask, :, :][0, 0, :]
    )).squeeze() + 1
    opt_duration = np.diff(opt_start_stop_bins)[0] / rasterfs

    # get only the relevant part of the raster,
    # using the light onset time as an anchor point
    start_time = prestim + tstart
    end_time = prestim + tend
    start_bin = np.floor(start_time * rasterfs).astype(int)
    end_bin = np.floor(end_time * rasterfs).astype(int)

    raw_raster = rec['resp'].extract_epoch(
        'REFERENCE'
    )[:, :, start_bin:end_bin]

    ON = raw_raster[opto_mask, 0, :]
    OFF = raw_raster[~opto_mask, 0, :]

    t = np.linspace(tstart, tend, end_bin - start_bin,
                    endpoint=False) * 1000  # in ms

    #### now plots ####
    fig = make_subplots(3, 1)

    for oo, (arr, color) in enumerate(zip([ON, OFF], [PHOTOACT, '#696969'])):
        # spike raster / light onset/offset
        yoffset = (ON.shape[0] - 1) * oo
        st = np.where(arr)
        x = ((st[1] / rasterfs) + tstart) * 1000
        _ = fig.add_trace(go.Scatter(x=x, y=st[0] + yoffset, mode='markers',
                                     marker=dict(color=color, size=1)),
                          row=1, col=1)

        # psth
        y = arr.mean(axis=0) * rasterfs / 1000
        _ = fig.add_trace(go.Scatter(x=t, y=y, mode='lines',
                                     line=dict(color=color, width=1)),
                          row=2, col=1)  # last row so it keeps the x axis

    # inset of waveform
    mean_waveform = toclust.query(f"id == '{cellid}'").waveform_norm.values[0]
    x = np.linspace(t[0], t[-1], len(mean_waveform))  # hack to share axes
    _ = fig.add_trace(go.Scatter(x=x, y=mean_waveform, mode='lines',
                                 line=dict(color=PHOTOACT, width=1)), row=3,
                      col=1)

    for row in [1, 2]:
        for vline in [0, opt_duration * 1000]:
            fig.add_vline(x=vline,
                          line=dict(width=1, dash='dash', color='black'),
                          opacity=1, row=row, col=1)

    # format
    w, h = 2, 2
    fig.update_layout(
        template='simple_white', width=96 * w, height=96 * h,
        margin=dict(t=10, b=10, l=10, r=10),

        xaxis=dict(matches='x2', autorange=False,
                   showticklabels=False, ),
        yaxis=dict(title=dict(standoff=0, text='trials', font_size=10),
                   domain=[0.5, 1], showticklabels=False, ticks='',
                   tickfont_size=9),
        xaxis2=dict(title=dict(standoff=0, text='time (ms)',
                               font_size=10), range=[-20, 100],
                    tickfont_size=9),
        yaxis2=dict(title=dict(text='firing rate (spk/ms)', font_size=10,
                               standoff=0), domain=[0, 0.5],
                    tickfont_size=9),
        # inset
        xaxis3=dict(matches=None, showticklabels=False, ticks='',
                    showline=False, domain=[0.5, 0.9]),
        yaxis3=dict(showticklabels=False, ticks='',
                    showline=False, domain=[0.2, 0.4]),

        showlegend=False,
        legend=dict(xanchor='right', x=1, yanchor='bottom', y=0,
                    font_size=9, title_text='',
                    bgcolor="rgba(0,0,0,0)")
    )

    return fig


def plot_metric_cumulative_histograms():
    """
    Figure 5 panels D and E. Displays cumulative histograms of Amplitude and
    Duration of context effects, colored by the different cell type
    classifications. It also prints a kruskal wallis nonparametric ANOVA
    comparing the metrics between cell types, and a post hoc Dunn test
    for pairwise comparisons.

    Returns: Plotly figure

    """

    # integral in seconds instead of ms
    toplot.loc[
        toplot.metric == 'integral', 'value'
    ] = toplot.loc[toplot.metric == 'integral', 'value'] / 1000

    toplot['triple'] = toplot.triple.cat.remove_unused_categories()

    fig = px.ecdf(
        toplot, x='value', color='triple', facet_col='metric',
        color_discrete_map={'activated': PHOTOACT,
                            'narrow': NAROWSPIKE,
                            'broad': BROADSPIKE},
        category_orders={'triple': ['activated', 'narrow', 'broad']},
        render_mode='svg'
    )
    _ = fig.update_traces(line_width=1)

    # formating
    w, h = 3, 2
    fig.update_layout(
        template='simple_white', width=96 * w, height=96 * h,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(xanchor='right', x=1, yanchor='bottom',
                    y=0.05, font_size=9, title=dict(text='')),
        xaxis=dict(matches=None, autorange=False, range=[0, 1],
                   title=dict(text='Amplitude (Delta Z-score*s)',
                              font_color=AMPCOLOR)),
        yaxis=dict(title=dict(text='proportion',
                              font_size=10,
                              standoff=0),
                   tickfont_size=9),
        xaxis2=dict(matches=None, autorange=True,
                    title=dict(text='Duration (ms)',
                               font_color=DURCOLOR))

    )
    # Common formatting of both x axes
    fig.update_xaxes(title=dict(font_size=10, standoff=0), tickfont_size=9)

    # stats
    for met in toplot.metric.unique():
        print(f"\n######## {met} ########")
        _ = kruskal_with_posthoc(toplot.query(f"metric == '{met}'"),
                                 group_col='triple', val_col='value')

    return fig

def plot_metric_insets():
    """
    Figure 5 Panels D and E, insets. Shows the median for the Amplitude and
    the mean for de Duration metrics, for each of the cell classifications.
    Also shows the jack-knifed 95% confidence interval of. This is a simplified
    visual representation of the Kurskal wallis and Dunn post hoc results from
    the main panels containing the insets.

    Returns: Plotly figure

    """
    # add someting along the line of means and SEMs
    summary = toplot.groupby(['metric', 'triple']).agg(
        median=('value', lambda x: jknf(x, 1000, np.nanmedian)[0]),
        SEmedian=('value', lambda x: jknf(x, 1000, np.nanmedian)[1]),
        mean=('value', lambda x: jknf(x, 1000, np.nanmean)[0]),
        SEmean=('value', lambda x: jknf(x, 1000, np.nanmean)[1]),
    )

    metrics = ['integral', 'last_bin']
    fig = make_subplots(1, 2, horizontal_spacing=0.2, subplot_titles=metrics)
    for cc, met in enumerate(metrics):
        met_df = summary.loc[(met), :].reindex(
            index=['activated', 'narrow', 'broad'])
        print(f"\n#### {met} bar value ####\n{met_df}")
        if met == 'integral':
            # Uses Median for Amplitude since it relate to cumulative
            # histograms and statistical tests
            stat = met_df['median'].values
            err = met_df['SEmedian'].values
            ticktext = [f'{m:.3f}' for m in stat]
        elif met == 'last_bin':
            # Since it takes discrete values, the median and CI look very
            # disturbing. Instead, Uses the mean. This is an entirely visual
            # choice, and the statistics computed for the outer panel still
            # hold.
            stat = met_df['mean'].values
            err = met_df['SEmean'].values
            ticktext = [f'{m:.1f}' for m in stat]
        names = met_df.index.tolist()
        # added line to this single call, make sure it actually works
        fig.add_trace(
            go.Scatter(x=stat, y=names, mode='markers', opacity=1,
                       showlegend=False,
                       marker=dict(color=[PHOTOACT, NAROWSPIKE, BROADSPIKE],
                                   symbol='square', size=5,
                                   line=dict(color='black',
                                             width=1)),
                       error_x=dict(array=err, color='black',
                                    thickness=1, width=10)),
            row=1, col=cc + 1
        )
        fig.update_xaxes(tickmode='array', tickvals=stat, ticktext=ticktext,
                         tickangle=-90, row=1, col=cc + 1)

    # formating
    w, h = 2, 1.3
    fig.update_layout(template='simple_white', width=96 * w, height=96 * h,
                      margin=dict(t=20, b=20, l=10, r=10), showlegend=False)

    # commong features of both x axes
    fig.update_yaxes(showticklabels=False, ticks='', showline=False,
                     title=dict(font_size=10, standoff=0), tickfont_size=9)

    fig.update_xaxes(showticklabels=True, ticks='outside', showline=True,
                     title=dict(font_size=10, standoff=0), tickfont_size=9)

    return fig


coverage_prop = toregress.query(
    "triple in ['activated', 'narrow', 'broad'] "
    "and metric == 'integral' "
    "and stim_count in [10]"
    # Only select this subset as is the one containing the optotagged neurons
).groupby(['triple', 'id'], observed=True).agg(
    coverage=('significant', lambda x: sum(x) / len(x) * 100),
    count=('significant', 'count')
).reset_index()


def plot_coverage_cumulative_histogram():
    """
    Figure 5 panel F. Displays cumulative histograms of context space coverage
    colored by the different cell type classifications. It also prints a
    kruskal wallis nonparametric ANOVA comparing the metrics between cell
    types, and a post hoc Dunn test for pairwise comparisons.

    Returns: Plotly figure

    """

    print("\n####### non parametric anova with posthoc #######")
    _ = kruskal_with_posthoc(coverage_prop, 'triple', 'coverage')

    fig = px.ecdf(
        coverage_prop, x='coverage', color='triple',
        color_discrete_map={'activated': PHOTOACT,
                            'narrow': NAROWSPIKE,
                            'broad': BROADSPIKE},
        category_orders={'triple': ['activated', 'narrow', 'broad']},
        render_mode='svg')
    _ = fig.update_traces(line_width=1)

    # formating
    w, h = 1.5, 2
    fig.update_layout(
        template='simple_white', width=96 * w, height=96 * h,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(xanchor='right', x=1, yanchor='bottom',
                    y=0.05, font_size=9, title=dict(text='')),
        xaxis=dict(matches=None, autorange=False, range=[0, 30],
                   title=dict(text='Contextual coverage %')),
        yaxis=dict(title=dict(text='proportion', font_size=10,
                              standoff=0), tickfont_size=9),

    )
    # commong features of both x axes
    fig.update_xaxes(title=dict(font_size=10, standoff=0), tickfont_size=9)

    return fig


def plot_coverage_inset():
    """
    Figure 5 Panel F inset. Shows the median context space coverage percent
    for each of the cell classifications.
    Also shows the jack-knifed 95% confidence interval of. This is a simplified
    visual representation of the Kurskal wallis and Dunn post hoc results from
    the main panels containing the insets.

    Returns: Plotly figure

    """
    coverage_summary = coverage_prop.groupby(['triple'], observed=True).agg(
        coverage=('coverage', lambda x: jknf(x, 100, np.median)[0]),
        error=('coverage', lambda x: jknf(x, 100, np.median)[1])
    ).reindex(index=['activated', 'narrow', 'broad'])

    print(f"\n##### Context coverage bar values #####\n{coverage_summary}")

    fig = go.Figure()

    stat = coverage_summary['coverage'].values
    err = coverage_summary['error'].values
    names = coverage_summary.index.tolist()
    # added line to this single call, make sure it actually works
    fig.add_trace(go.Scatter(x=stat, y=names, mode='markers', opacity=1,
                             showlegend=False,
                             marker=dict(color=[PHOTOACT,
                                                NAROWSPIKE,
                                                BROADSPIKE],
                                         symbol='square', size=5,
                                         line=dict(color='black', width=1)),
                             error_x=dict(array=err, color='black',
                                          thickness=1, width=10), ))

    fig.update_xaxes(tickmode='array', tickvals=stat,
                     ticktext=[f'{m:.3f}' for m in stat], tickangle=-90)

    # formating
    w, h = 1, 1.3
    fig.update_layout(template='simple_white', width=96 * w, height=96 * h,
                      margin=dict(t=20, b=20, l=10, r=10), showlegend=False)

    # commong features of both x axes
    fig.update_yaxes(showticklabels=False, ticks='', showline=False,
                     title=dict(font_size=10, standoff=0), tickfont_size=9)

    fig.update_xaxes(showticklabels=True, ticks='outside', showline=True,
                     title=dict(font_size=10, standoff=0), tickfont_size=9)

    return fig
