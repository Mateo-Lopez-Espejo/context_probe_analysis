import itertools as itt
import pathlib as pl

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import wavfile
from scipy.stats import linregress, sem, wilcoxon, norm
from sklearn.metrics import r2_score

from nems.analysis.gammatone.gtgram import gtgram
from publication.globals import (
    DISPLAY_NAME_MAP, MODEL_CTX_QUANT_DF, MODEL_PERFORMANCE_WIDE_DF,
    MODEL_STATISTIC, MODEL_DISPLAY_NAMES, MODEL_NICKNAMES
)
from src.models.modelnames import modelnames as all_modelnames
from src.models.param_tools import get_population_weights, get_strf
from src.utils.subsets import batch_map
from src.utils.tools import decimate_xy
from src.visualization.interactive import plot_raw_pair
from src.visualization.palette import MODEL_COLORMAP, TENCOLOR


def plot_encoding_model_schematic():
    """
    Figure 6 Panel A. General model architecture schematic base. Displays a
    spectrogram, and an STRF as a kernel rolling over it. Also displays
    a cartoon of a neuronal population response with the respective integration
    windows, the average and the weight applied.

    Returns: Plotly figure

    """
    # Use simulation if no access to real data. Not a big deal
    # since it's just a cartoon after all
    sim = True

    def heatmap_with_margin(**kwargs):
        if 'x0' in kwargs.keys():
            x0 = kwargs['x0'] - 0.5
            y0 = kwargs['y0'] - 0.5
            dy, dx = kwargs['z'].shape
        else:
            x0 = kwargs['x'][0]
            dx = kwargs['x'][-1] - kwargs['x'][0]
            y0 = kwargs['y'][0]
            dy = kwargs['y'][-1] - kwargs['y'][0]
        traces = list()
        traces.append(go.Heatmap(**kwargs))
        traces.append(go.Scatter(x=[x0, x0, x0 + dx, x0 + dx, x0],
                                 y=[y0, y0 + dy, y0 + dy, y0, y0],
                                 mode='lines', showlegend=False,
                                 line=dict(color='black',
                                           width=1)
                                 )
                      )
        return traces

    def poissonSpikeGen(fr, binSim, nTrials):
        tSim = binSim / 100  # 100 hz resolution for models
        dt = 1 / 1000  # ms resolution for nicer plots
        nBins = np.floor(tSim / dt).astype(int)
        spikeMat = np.random.random((nTrials, nBins)) < fr * dt
        tVec = np.linspace(0, binSim, nBins, endpoint=False)
        return spikeMat, tVec

    # creates some fake data
    nneur = 6  # other neurons in the population
    extra = 10

    # Get Self and Pop weights
    if sim:
        all_weights = np.random.random((nneur + 1)) * 2 - 1  # random weights
    else:
        eg_cellid = 'TNC014a-22-2'
        eg_modelname = all_modelnames['matchl_full']
        all_weights = get_population_weights(
            eg_cellid, batch_map[eg_cellid], eg_modelname
        )

    pop_weights = all_weights[1:nneur + 1, np.newaxis]
    self_weight = all_weights[[0], np.newaxis]

    # get STRF weights
    if sim:
        strf = np.random.random((18, 30))  # random STRF
    else:
        strf = get_strf(eg_cellid, batch_map[eg_cellid], eg_modelname)

    dur = strf.shape[1]

    # simulate some random spikes of neuron activity
    PSTHs, t_psth = poissonSpikeGen(fr=40, binSim=extra + int(dur / 2),
                                    nTrials=nneur)
    PSTHs_mean = PSTHs.mean(axis=1, keepdims=True)

    # simulated neuron
    resp_psth, t_resp = poissonSpikeGen(fr=40, binSim=extra + dur,
                                        nTrials=1)  # one neu by time
    resp_mean = resp_psth.mean(axis=1, keepdims=True)
    resp_psth = resp_psth.squeeze()

    #### Spectrogram ####
    if sim:
        spectrogram = np.random.random((18, strf.shape[1] + extra))
    else:
        path = pl.Path(
            "/auto/users/mateo/code/baphy/Config/lbhb/SoundObjects/"
            "@NaturalPairs/NatPairSounds/"
            "07_cat220_rec1_latin-pop_enrique-iglesias_"
            "i-will-survive_excerpt1_7.wav")
        sfs, W = wavfile.read(path)
        spectrogram = gtgram(W, sfs, 0.02, 0.01, 48, 100, 24000)
        pass

    # parameters defining the positions of schematic elements
    psth_scaling = 1.5
    STRF_vert_offset = 5  # relative to the self PSTH
    self_PSTH_vert_offset = 1  # relative to the last pop_psth
    STRF_float = 0.5
    cell_avg_lpad = 4
    cell_weighs_lpad = 4

    fig = go.Figure()

    # add PSTHS for other neurons in the population
    for ll, psth in enumerate(PSTHs):
        y = psth * psth_scaling + ll
        x = np.linspace(0, spectrogram.shape[1] - int(strf.shape[1] / 2),
                        psth.shape[0], endpoint=False)

        fig.add_trace(
            go.Scatter(x=x, y=y,
                       mode='lines',
                       line=dict(
                           color='black',
                           width=1
                       ),
                       showlegend=False
                       )
        )

    pop_ytickvals = list(range(nneur - 1, -1, -1))
    pop_yticktext = [f'neighbor neuron {ii + 1}' if ii == 0 else f'{ii + 1}'
                     for ii, _ in enumerate(pop_ytickvals)]

    # Add PSTH for the predicted neuron
    y = resp_psth * psth_scaling + nneur + self_PSTH_vert_offset
    x = np.linspace(0, spectrogram.shape[1], resp_psth.shape[0],
                    endpoint=False)
    fig.add_trace(
        go.Scatter(x=x, y=y,
                   mode='lines',
                   opacity=1,
                   line=dict(
                       color='black',
                       width=1.5
                   ),
                   showlegend=False
                   )
    )

    resp_ytickvals = [nneur + self_PSTH_vert_offset]
    resp_yticktext = [f'target neuron']

    heamaps_y0 = nneur + self_PSTH_vert_offset + STRF_vert_offset

    # Sound spectrogram
    hx = np.linspace(0.5, dur + extra, spectrogram.shape[1], endpoint=False)
    hy = np.linspace(heamaps_y0, heamaps_y0 + strf.shape[0],
                     spectrogram.shape[0], endpoint=False)
    fig.add_traces(heatmap_with_margin(z=spectrogram, x=hx, y=hy,
                                       showscale=False,
                                       coloraxis='coloraxis2',
                                       opacity=1,
                                       zsmooth='best', ))

    # STRF
    strf_y0 = heamaps_y0 - STRF_float
    fig.add_traces(heatmap_with_margin(z=strf[:, ::-1],
                                       x0=extra + 0.5,
                                       dx=1,
                                       y0=strf_y0,
                                       dy=1,
                                       showscale=False,
                                       coloraxis='coloraxis',
                                       opacity=1,
                                       zsmooth='best',
                                       zmin=-1, zmid=0, zmax=1
                                       )
                   )

    # ticks every second freq bin??
    all_freqs = 2 ** np.linspace(0, spectrogram.shape[0] * 1 / 3,
                                 spectrogram.shape[0], endpoint=False) * 200
    strf_ytickvals = list(range(heamaps_y0, heamaps_y0 + strf.shape[0], 2))
    strf_yticktext = [f'{int(frq)} hz' for ii, frq in enumerate(all_freqs) if
                      ii % 2 == 0]

    # average of population
    x0 = extra + int(dur / 2) + cell_avg_lpad
    dx = 1
    y0 = 0
    dy = 1
    fig.add_traces(heatmap_with_margin(z=PSTHs_mean,
                                       x0=x0, dx=dx,
                                       y0=y0, dy=dy,
                                       showscale=False,
                                       coloraxis='coloraxis4')
                   )

    # population weight
    x0 = extra + int(dur / 2) + cell_avg_lpad + cell_weighs_lpad
    fig.add_traces(heatmap_with_margin(z=pop_weights,
                                       x0=x0, dx=dx,
                                       y0=y0, dy=dy,
                                       showscale=False,
                                       coloraxis='coloraxis3')
                   )

    # average of self
    x0 = extra + int(dur / 2) + cell_avg_lpad
    dx = 1
    y0 = nneur + self_PSTH_vert_offset
    dy = 1
    fig.add_traces(heatmap_with_margin(z=resp_mean,
                                       x0=x0, dx=dx,
                                       y0=y0, dy=dy,
                                       showscale=False,
                                       coloraxis='coloraxis4')
                   )

    # self weight
    x0 = extra + int(dur / 2) + cell_avg_lpad + cell_weighs_lpad
    fig.add_traces(heatmap_with_margin(z=self_weight,
                                       x0=x0, dx=dx,
                                       y0=y0, dy=dy,
                                       showscale=False,
                                       coloraxis='coloraxis3')
                   )

    # box around all PSTHs
    x0 = extra
    dx = int(strf.shape[1] / 2)
    x = np.array([x0, x0 + dx, x0 + dx, x0, x0])
    y0 = -0.5
    dy = nneur + self_PSTH_vert_offset + psth_scaling * 2
    y = np.array([y0, y0, y0 + dy, y0 + dy, y0]) - 0.5
    fig.add_trace(go.Scatter(x=x, y=y,
                             mode='lines',
                             line=dict(color='black',
                                       dash='solid',
                                       width=1),
                             showlegend=False,
                             fill='toself', fillcolor='rgba(256,256,256,0.7)'
                             ))

    ytickvals = pop_ytickvals + resp_ytickvals + strf_ytickvals
    yticktext = pop_yticktext + resp_yticktext + strf_yticktext

    w, h = 5, 3
    fig.update_layout(
        width=w * 96, height=h * 96,
        margin=dict(b=10, t=10, l=10, r=10),
        template='simple_white',
        coloraxis=dict(
            showscale=True,
            colorscale='RdBu',
            cmid=0,
            reversescale=True,
            colorbar=dict(
                thickness=5, len=0.6,
                title=dict(text='model weights (au)',
                           font_size=10,
                           side='right'),
                tickfont_size=9,
                tickangle=-90,
                xanchor='left',
                x=1,
                yanchor='top',
                y=1
            )
        ),

        coloraxis2=dict(showscale=False,
                        colorscale='greys',
                        reversescale=False),
        coloraxis3=dict(showscale=False,
                        colorscale='RdBu',
                        cmid=0,
                        reversescale=True),

        coloraxis4=dict(showscale=False,
                        colorscale='greys',
                        reversescale=True),

        xaxis=dict(autorange=True,
                   constrain='domain',
                   tickmode='array',
                   tickvals=np.asarray(
                       [0, 5, 10, 15, 20, 25, 30]) + extra,
                   ticktext=[-300, -250, -200, -150, -100, -50,
                             0],
                   tickfont_size=9,
                   title=dict(text='time from prediction (ms)',
                              standoff=0,
                              font_size=10)),

        yaxis=dict(scaleanchor='x',
                   ticks='',
                   tickmode='array',
                   tickvals=ytickvals,
                   ticktext=yticktext,
                   tickfont_size=9,
                   showline=False,
                   constrain='domain')
    )
    return fig


def plot_example_model_predictions():
    """
    Figure 6 Panel B. Shows example PSTH of one neuron one probe and two
    contexts. The PSTHs are for the neuron response and for two models
    predictions. The difference between the contexts i.e. the context effect
    is highlighted.

    Returns: Plotly figure

    """
    eg_cellid, eg_contexts, eg_probe = 'TNC014a-22-2', (0, 8), 3

    # True stands for prediction, False for response
    combos = [
        (False, 'matchl_STRF'),
        (True, 'matchl_STRF'),
        # (True,'matchl_self'),
        # (True,'matchl_pop'),
        (True, 'matchl_full'),
    ]

    fig = make_subplots(1, len(combos), shared_xaxes='all', shared_yaxes='all',
                        horizontal_spacing=0.02)

    for cc, (plot_pred, modname) in enumerate(combos):
        if cc == 0:
            sw = 30
        else:
            sw = 0
        f = plot_raw_pair(eg_cellid, eg_contexts, eg_probe,
                          modelname=all_modelnames[modname],
                          batch=batch_map[eg_cellid],
                          part='probe',
                          plot_pred=plot_pred,
                          hightlight_difference=True,
                          raster_fs=100, smoothing_window=sw,
                          colors=TENCOLOR)
        f.update_traces(line_width=1)
        if cc >= 2:
            f.update_traces(showlegend=False)
        t = f['data']
        fig.add_traces(t, cols=[cc + 1] * len(t), rows=[1] * len(t))

        # colored names for the STRFs
        if cc > 0:
            fig.add_trace(go.Scatter(x=[0.5], y=[0.5],
                                     text=[
                                         f'<b>{DISPLAY_NAME_MAP[modname]}</b>'
                                     ],
                                     mode='text', showlegend=False,
                                     textfont=dict(size=12,
                                                   color=MODEL_COLORMAP[
                                                       modname]),
                                     textposition='bottom center'
                                     ),
                          col=cc + 1, row=1)

    w, h = 5, 2  # in inches
    _ = fig.update_layout(
        template='simple_white',
        margin=dict(l=10, r=10, t=10, b=10),
        width=round(96 * w), height=round(96 * h),
        legend=dict(x=1 / 3, y=1, font=dict(size=9),
                    bgcolor="rgba(0,0,0,0)")
    )

    for vline in [0.25, 0.5, 0.75]:
        fig.add_vline(x=vline, line=dict(width=1, dash='dot', color='black'))

    fig.update_xaxes(title=dict(text='time from probe onset (s)',
                                font_size=10, standoff=0),
                     tickfont_size=9)
    fig.update_yaxes(title=dict(text='normalized firing rate',
                                font_size=10, standoff=0),
                     tickfont_size=9, col=1)

    return fig


def _print_pairwise_model_comparisons():
    print(
        "#### summary metrics ####\n",
        MODEL_PERFORMANCE_WIDE_DF.loc[:, MODEL_DISPLAY_NAMES].agg(
            ['mean', 'sem', 'std'])
    )

    # calculate pairwise statistical comparisons
    print("\n#### pairwise statistical tests ####\n")
    for m0, m1 in itt.combinations(MODEL_DISPLAY_NAMES, r=2):
        out = wilcoxon(MODEL_PERFORMANCE_WIDE_DF[m0],
                       MODEL_PERFORMANCE_WIDE_DF[m1])
        print(f'Wilcoxon: {m0}vs{m1}, pvalue={out.pvalue}')
    model_count = MODEL_PERFORMANCE_WIDE_DF.groupby('region',
                                                    observed=True).agg(
        count=('id', pd.Series.nunique))
    print(
        "#### pairwise statistical comparisons ####\n",
        model_count
    )
    print(
        "#### model count ####\n",
        model_count.sum()
    )
    return None


def plot_relative_model_performances():
    """
    Figure 6 Panel C. Plots the overall models performances as
    the response/prediction Pearson's R, excluding those models with
    performance lower than chance. The performances are shown relative to the
    Full (best) model. This performance calculation is agnostic to
    context effects. Shows all neurons for every model as swarm plots
    with lines indicating the relationship of different models for the same
    neurons. It also displays the mean and SEM for each model.

    Prints information of the ABSOLUTE (not relative to Full) mean and SEM
    values and the pairwise Wilconxon paired test between every model.
    The results of these tests are used to define significance symbols
    in the manuscript final figure.

    Returns: Plotly figure.

    """

    ref_mod = 'matchl_full'  # reference value to use as ceiling

    _print_pairwise_model_comparisons()

    norm_wide = MODEL_PERFORMANCE_WIDE_DF.copy()
    # replace fluke  negative r-test with zeros
    for col in ['STRF', 'Self', 'Pop', 'Full']:
        norm_wide.loc[norm_wide[col] < 0, col] = 0

    for col in MODEL_DISPLAY_NAMES:
        norm_wide[col] = norm_wide[col] / norm_wide[DISPLAY_NAME_MAP[ref_mod]]

    # Outlier with the best performance not in the Full model, likely due
    # to fit noise.
    # Filters them from plots for better visualization, i.e. to drive to point
    # of relative model performances. But keeps them for statistical tests
    # on the plot showing the absolute model performances
    print(
        "\n#### removing from plot visual outliers "
        "but keeping for statistics ####\n",
        norm_wide.query("STRF > 1 and Self > 1 and Pop > 1")
    )
    norm_wide.query("STRF < 1", inplace=True)

    toplot_norm = pd.melt(
        norm_wide, id_vars=['id', 'site', 'region'],
        value_vars=MODEL_DISPLAY_NAMES,
        var_name='model', value_name=MODEL_STATISTIC
    )

    fig = go.Figure()

    # single lines grayed out
    for ll in norm_wide.loc[:, MODEL_DISPLAY_NAMES].values:
        fig.add_trace(
            go.Scatter(
                x=MODEL_DISPLAY_NAMES, y=ll, mode='lines', opacity=0.1,
                line=dict(color='gray', width=1), showlegend=False
            )
        )

    # mean and SEM line+markers
    mean = norm_wide.loc[:, MODEL_DISPLAY_NAMES].values.mean(axis=0)
    err = sem(norm_wide.loc[:, MODEL_DISPLAY_NAMES].values, axis=0)

    # added line to this single call, make sure it actually works
    fig.add_trace(
        go.Scatter(
            x=MODEL_DISPLAY_NAMES, y=mean,
            mode='markers+lines',
            opacity=1,
            showlegend=False,
            marker=dict(color=[MODEL_COLORMAP[nknm] for nknm in
                               MODEL_DISPLAY_NAMES],
                        symbol='square',
                        size=5,
                        line=dict(color='black',
                                  width=1)),
            error_y=dict(array=err,
                         color='black',
                         thickness=1,
                         width=10),
            line=dict(color='black')
        )
    )

    # individual dots
    strip = px.strip(
        toplot_norm, y=MODEL_STATISTIC, x='model', color='model',
        color_discrete_map={DISPLAY_NAME_MAP[nknm]: clr for nknm, clr
                            in MODEL_COLORMAP.items()}
    )
    strip = strip.update_traces(marker=dict(opacity=1,
                                            size=2)
                                )['data']
    fig.add_traces(strip)

    w, h = 2, 2  # in inches
    _ = fig.update_layout(
        template='simple_white',
        margin=dict(l=10, r=10, t=10, b=10),
        width=round(96 * w), height=round(96 * h),
        xaxis1=dict(domain=[0, 1], anchor='y1',
                    title_text='model'),
        yaxis1=dict(domain=[0, 1], anchor='x1',
                    # range=[0, 3],
                    title_text='r-test fraction of best'),
        showlegend=False,
        # background color changes for all subplots,
        plot_bgcolor='rgba(256,256,256,0.2)'
    )

    fig.update_xaxes(title=dict(font_size=10, standoff=0), tickfont_size=9)
    fig.update_yaxes(title=dict(font_size=10, standoff=0), tickfont_size=9)

    return fig


def plot_absolute_model_performances():
    """
    Supplementary figure X panel Y. Plots the overall models performances as
    the response/prediction Pearson's R, excluding those models with
    performance lower than chance. This performance calculation is agnostic to
    context effects. Shows all neurons for every model as swarm plots
    with lines indicating the relationship of different models for the same
    neurons. It also displays the mean and SEM for each model, and adds an
    inset showing the same mean and SEM without the individual data paints for
    clarity. Prints the mean and SEM values and the pairwise Wilconxon paired
    test between every model. The results of these tests are used to define
    significance symbols in the manuscript final figure which uses the sister
    figure of this one, comparing the relative model performances:
    see 'plot_relative_model_performances'.

    Returns: Plotly figure.

    """

    _print_pairwise_model_comparisons()

    # transforms wide into long format for plotly compatibility
    toplot = pd.melt(
        MODEL_PERFORMANCE_WIDE_DF,
        id_vars=['id', 'site', 'region'],
        value_vars=MODEL_DISPLAY_NAMES, var_name='model',
        value_name=MODEL_STATISTIC
    )

    fig = make_subplots(cols=1, rows=2, shared_xaxes=False)

    # Single models grayed out lines
    for ll in MODEL_PERFORMANCE_WIDE_DF.loc[:, MODEL_DISPLAY_NAMES].values:
        fig.add_trace(
            go.Scatter(x=MODEL_DISPLAY_NAMES, y=ll,
                       mode='lines',
                       opacity=0.1,
                       line=dict(
                           color='gray',
                           width=1
                       ),
                       showlegend=False
                       ),
            row=1, col=1
        )

    # individual dots
    strip = px.strip(
        toplot, y=MODEL_STATISTIC, x='model', color='model',
        color_discrete_map={DISPLAY_NAME_MAP[nknm]: clr for
                            nknm, clr in
                            MODEL_COLORMAP.items()},
    ).update_traces(
        marker=dict(opacity=1, size=2)
    )['data']
    fig.add_traces(strip, rows=[1] * len(strip), cols=[1] * len(strip))

    # mean and SEM line+markers
    mean = MODEL_PERFORMANCE_WIDE_DF.loc[:, MODEL_DISPLAY_NAMES].values.mean(
        axis=0)
    err = sem(MODEL_PERFORMANCE_WIDE_DF.loc[:, MODEL_DISPLAY_NAMES].values,
              axis=0)

    mean_markers = go.Scatter(
        x=MODEL_DISPLAY_NAMES, y=mean,
        mode='markers+lines',
        opacity=1,
        showlegend=False,
        marker=dict(
            color=[MODEL_COLORMAP[nknm] for nknm in
                   MODEL_NICKNAMES],
            symbol='square',
            size=5,
            line=dict(color='black',
                      width=1)),
        error_y=dict(array=err,
                     color='black',
                     thickness=1,
                     width=10),
        line=dict(color='black')
    )

    # add mean markers both on the main panel and oin the instet
    fig.add_trace(mean_markers, row=1, col=1)  # main
    fig.add_trace(mean_markers, row=1, col=2)  # inset

    w, h = 2.5, 3.5  # in inches
    _ = fig.update_layout(
        template='simple_white',
        margin=dict(l=10, r=10, t=10, b=10),
        width=round(96 * w), height=round(96 * h),
        xaxis1=dict(domain=[0, 1], anchor='y1',
                    title_text='model'),
        yaxis1=dict(domain=[0, 1], anchor='x1',
                    range=[0, 1],
                    title_text='r-test'),
        xaxis2=dict(domain=[0.15, 0.35], anchor='y2',
                    ticks='', showticklabels=False),
        yaxis2=dict(domain=[0.7, 1], anchor='x2'),
        showlegend=False,
        # background color changes for all subplots,
        plot_bgcolor='rgba(256,256,256,0.2)'
    )

    fig.update_xaxes(title=dict(font_size=10, standoff=0), tickfont_size=9)
    fig.update_yaxes(title=dict(font_size=10, standoff=0), tickfont_size=9)

    return fig


def plot_context_effect_prediction_scatter():
    """
    Figure 6 panel D. Displays the context effects calculated for the recorded
    data vs the model predictions, fits a linear regression to the scatter and
    print the regression coefficients. Does this for the STRF (worse, base)
    and the Full (best) models; and for 3 selected time periods A to C
    displayed as i to iii in the manuscript final figure.

    Returns: Plotly figure

    """

    decimate = 1000
    time_bins = ['A', 'B', 'C']
    selected = ['matchl_STRF', 'matchl_full']

    # test plotting please ignore
    fig = make_subplots(rows=1, cols=len(time_bins), shared_yaxes='all',
                        shared_xaxes='all',
                        horizontal_spacing=0.02, vertical_spacing=0.01)

    print("\n#### measured vs predicted context effects correlation ####")
    for tt, tbin in enumerate(time_bins):
        ranges = list()
        for nn, nknm in enumerate(selected):
            subplotdf = MODEL_CTX_QUANT_DF.query(
                f"metric == 'integral' "
                f"and time_bin == '{tbin}' "
                f"and  nickname == '{nknm}'"
            )

            x = subplotdf.response.values.astype(float)
            y = subplotdf.value.values.astype(float)
            R2 = r2_score(x, y)
            slope, bias, r, pval, _ = linregress(x, y)
            print(f"{tbin}, {nknm}, R2={R2}, r={r}, pvalue={pval}")

            xm, ym = decimate_xy(x, y, decimate, by_quantiles=False)

            fig.add_trace(
                go.Scatter(x=xm, y=ym, mode='markers',
                           opacity=0.8,
                           marker=dict(
                               opacity=0.8,
                               color=MODEL_COLORMAP[nknm],
                               size=2
                           ),
                           name=f'{nknm}',
                           hoverinfo='skip', showlegend=False
                           ),
                col=tt + 1, row=1
            )

            model_range = [np.nanmin(np.stack([xm, ym])),
                           np.nanmax(np.stack([xm, ym]))]

            # regression lines
            fig.add_trace(
                go.Scatter(x=model_range,
                           y=np.asarray(model_range) * slope + bias,
                           mode='lines',
                           line=dict(
                               color=MODEL_COLORMAP[nknm],
                               width=1
                           ),
                           opacity=1,
                           name=f'{nknm}',
                           hoverinfo='skip',
                           showlegend=False
                           ),
                col=tt + 1, row=1
            )

            ranges.append(model_range)

        # unity line
        ranges = np.stack(ranges, axis=1)
        plot_range = [ranges[0, :].min(), ranges[1, :].max()]
        plot_range = [plot_range[0] - (np.diff(plot_range)[0] * 0.025),
                      plot_range[1] + (np.diff(plot_range)[0] * 0.025)]

        fig.add_trace(
            go.Scatter(x=plot_range, y=plot_range, mode='lines',
                       line=dict(color='black',
                                 dash='dot',
                                 width=1),
                       showlegend=False),
            col=tt + 1, row=1
        )

        # Interval annotations
        prange = plot_range[1] - plot_range[0]
        x = plot_range[0] + prange * 0.25
        fig.add_trace(
            go.Scatter(x=[x], y=[100],
                       text=[tbin],
                       mode='text',
                       textposition='bottom center',
                       textfont_size=10,
                       showlegend=False),
            col=tt + 1, row=1
        )

    w, h = 5, 2
    fig.update_layout(template='simple_white',
                      width=96 * w, height=96 * h,
                      margin=dict(l=10, r=10, t=10, b=10))

    fig.update_xaxes(autorange=True, constrain='domain',
                     title=dict(text='measured difference', font_size=10,
                                standoff=0),
                     tickfont_size=9)
    fig.update_yaxes(scaleanchor='x', constrain='domain',
                     tickfont_size=9)
    fig.update_yaxes(
        title=dict(text='predicted difference', font_size=10, standoff=0),
        col=1, row=1)

    return fig


def plot_summary_model_performance_bars():
    """
    Figure 6 panel E. Summary showing the models performances at capturing
    context effects as the mean correlation coefficients between recorded and
    predicted context effects. This is calculated at 4 time intervals
    A-D, or ii to iv in the manuscript final figure.
    The plot shows the mean Pearson's r and a confidence interval calculated
    with subsampling (jackknife). The function also plots a dataframe of
    pairwise statistical comparisons between models using a T-test , done
    independently for each time interval, and used to draw the significance
    symbols in the manuscript final figure.

    Returns: Plotly figure

    """

    # calculates pearsons correlation between real and predicted integral
    # values, for all combinations of models and time bin chunks

    def jknf(x, y, njacks=20):
        pred = x
        resp = y
        chunksize = int(np.ceil(len(pred) / njacks / 10))
        chunkcount = int(np.ceil(len(pred) / chunksize / njacks))
        idx = np.zeros((chunkcount, njacks, chunksize))
        for jj in range(njacks):
            idx[:, jj, :] = jj
        idx = np.reshape(idx, [-1])[:len(pred)]
        jc = np.zeros(njacks)
        for jj in range(njacks):
            ff = (idx != jj)
            jc[jj] = np.corrcoef(pred[ff], resp[ff])[0, 1]

        r_val = np.nanmean(jc)
        e = np.nanstd(jc) * np.sqrt(njacks - 1)
        var = np.nanmean((jc - r_val) ** 2) * (njacks - 1)
        return r_val, var, e

    to_regress = MODEL_CTX_QUANT_DF.query(f"metric == 'integral'").loc[:,
                 ['time_bin', 'nickname', 'response', 'value']]

    reg_df = to_regress.groupby(['time_bin', 'nickname'], observed=True).apply(
        lambda x: jknf(
            x['response'].values, x['value'].values, 20))

    reg_df = pd.DataFrame(reg_df.tolist(), index=reg_df.index,
                          columns=['pearsonsr', 'var', 'std'])

    print(
        "\n#### Number of Neurons Modeled ####\n",
        len(MODEL_CTX_QUANT_DF['id'].unique()),
        "\n#### Number of regressed instances ####\n",
        MODEL_CTX_QUANT_DF.query(
            f"metric == 'integral' "
            f"and time_bin == 'A' "
            f"and nickname == 'matchl_STRF'"
        ).shape[0],
    )
    # Number of models analyzed for regression

    # Pairwise comparisons between all bars within each time chunk
    pair_stats = list()
    for tb in ['A', 'B', 'C', 'D']:
        for nck1, nck2 in itt.combinations(
                reg_df.index.unique(level='nickname'), r=2):
            Q = (reg_df.loc[(tb, nck1), 'pearsonsr'] - reg_df.loc[
                (tb, nck2), 'pearsonsr']) / \
                np.sqrt(reg_df.loc[(tb, nck1), 'var'] + reg_df.loc[
                    (tb, nck2), 'var'])
            pval = norm(0, 1).pdf(Q)

            d = dict(time_bin=tb, model_1=nck1, model_2=nck2, stat=Q,
                     pvalue=pval)
            pair_stats.append(d)
    pair_stats = pd.DataFrame(pair_stats).set_index(
        ['time_bin', 'model_1', 'model_2'])

    pair_stats['corr_pval'] = pair_stats['pvalue'] * 6
    pair_stats['0.05'] = pair_stats['corr_pval'] < 0.05
    pair_stats['0.01'] = pair_stats['corr_pval'] < 0.01
    pair_stats['0.001'] = pair_stats['corr_pval'] < 0.001
    print(
        "\n#### paiwise statistical test within time bins ####\n",
        pair_stats
    )

    # plotting
    toplot = reg_df.reset_index().query(f"time_bin != 'full'")
    toplot['display_name'] = toplot.nickname.apply(
        lambda x: DISPLAY_NAME_MAP[x])
    DP_COLORMAP = {DISPLAY_NAME_MAP[nknm]: MODEL_COLORMAP[nknm] for nknm in
                   DISPLAY_NAME_MAP.keys()}

    # creates and stores panel
    fig = px.bar(data_frame=toplot, x='time_bin', y='pearsonsr',
                 color='display_name',
                 error_y='std', error_y_minus=None,
                 barmode='group',
                 category_orders={'time_bin': ['full', 'A', 'B', 'C', 'D'],
                                  'display_name': ['STRF',
                                                   'Self',
                                                   'Pop',
                                                   'Full']},
                 color_discrete_map=DP_COLORMAP,
                 )

    fig.update_traces(error_y=dict(thickness=1,
                                   width=2),
                      )

    w, h = 2, 2
    fig.update_layout(template='simple_white',
                      width=96 * w, height=96 * h,
                      margin=dict(l=10, r=10, t=10, b=10),
                      showlegend=False,
                      legend=dict(
                          orientation="h",
                          yanchor="bottom", y=1,
                          xanchor="left", x=0,
                          font_size=9
                      ))
    fig.update_xaxes(
        title=dict(text="time interval", font_size=10, standoff=0),
        tickfont_size=9)
    fig.update_yaxes(range=[0.3, 0.75],
                     title=dict(text="Pearson's R", font_size=10, standoff=0),
                     tickfont_size=9)

    return fig
