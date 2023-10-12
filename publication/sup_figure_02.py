"""
Functions related to pupil data frame preprocessing and plotting
"""
import itertools as itt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as sst
from plotly.subplots import make_subplots

from publication.globals import RASTER_META, ferret_df
from publication.globals import SUP_FIG3_WDF
from src.data.rasters import load_site_formated_raster
from src.root_path import root_path
from src.utils.dataframes import kruskal_with_posthoc
from src.utils.subsets import good_sites
from src.utils.tools import decimate_xy
from src.visualization.interactive import plot_raw_pair, plot_pupil_so_effects
from src.visualization.palette import Black, add_opacity
from src.visualization.palette import TENCOLOR

short_name_map = {
    'ferret fights Athena Violet001': 'ferret fight 1',
    'colouring freesound 123jorre456': 'coloring',
    'flute bourree': 'flute',
    'pop kelly clarkson einstein': 'USA pop',
    'latin pop enrique iglesias i will survive': 'latin pop',
    'fiddle luke abbott willie moore': 'fiddle',
    'ferret fights Jasmine Violet001': 'ferret fight 2'
}


def _organize_pupil_effects_by_sounds():
    # loads arrays of pupil time series, takes the probe part,
    # takes the averate over time, and asociates with the sound
    # preceding and simultaneous (context and probe) into a dataframe

    pupil_DF = list()
    for site in good_sites:
        if load_site_formated_raster.check_call_in_cache(
                site=site, part='all', pupil=True, **RASTER_META
        ):
            pupil, _ = load_site_formated_raster(
                site=site, part='all', pupil=True, **RASTER_META
            )
        else:
            print(f'no cache for site {site}')
            continue

        rep, chn, ctx, prb, tme = pupil.shape

        # number/names for context and probes. no silence (zero) as probe
        ctx_nms = range(0, ctx)
        prb_nms = range(1, prb + 1)

        # averages trial and removes cell unused singleton dimension
        R = pupil.mean(axis=0).squeeze(axis=0)

        half_bin = int(tme / 2)  # defines transition between context and probe
        part_dict = {
            'context': np.s_[..., :half_bin],
            'probe': np.s_[..., half_bin:]
        }
        df = list()
        for part_name, part_slicer in part_dict.items():
            part_R = R[part_slicer]  # shape chn x prb x tme

            for (cc, ctx_nm), (pp, prb_nm) in itt.product(
                    enumerate(ctx_nms), enumerate(prb_nms)
            ):
                d = {'site': site,
                     'part': part_name,
                     'context': ctx_nm,
                     'probe': prb_nm,
                     'value': np.mean(part_R[cc, pp])}
                df.append(d)

        pupil_DF.append(pd.DataFrame(df))

    pupil_DF = pd.concat(pupil_DF, ignore_index=True)

    # Sounds are identified by numbers, unique for each recording
    # relate the actual sound names

    # Loands and formats dataframe containing site sound names and numbers
    ferret_df['site'] = ferret_df['id'].str[:7]
    sound_name_DF = ferret_df.loc[
                    :, ('site', 'probe', 'named_probe')
                    ].drop_duplicates()

    # renames for clearer merge later on
    sound_name_DF.rename(columns={'probe': 'number',
                                  'named_probe': 'name'},
                         inplace=True)

    # add silence for all sites
    silence = pd.DataFrame(
        [sound_name_DF.site.unique()], index=('site',)
    ).T
    silence['number'] = 0
    silence['name'] = 'silence'

    # tidy format
    sound_name_DF = pd.concat(
        [sound_name_DF, silence]
    ).sort_values(
        ['site', 'number'], ignore_index=True
    )
    # formats sound names without - or _
    sound_name_DF['name'] = sound_name_DF['name'].str.replace(
        pat=r'(_|-)', repl=' ', regex=True
    )

    # Merges the sound names to initial pupil dataframe
    # names of probes
    DF = pd.merge(
        left=pupil_DF,
        right=sound_name_DF.rename(
            columns={'number': 'probe', 'name': 'probe_name'}),
        on=('site', 'probe'),
    )

    # names of contexts
    DF = pd.merge(
        left=DF,
        right=sound_name_DF.rename(
            columns={'number': 'context', 'name': 'context_name'}),
        on=('site', 'context'),
        validate='m:1'
    )

    # Use simplified names of the sounds
    DF = DF.replace(
        {'context_name': short_name_map, 'probe_name': short_name_map}
    )

    return DF


pupil_by_sound_DF = _organize_pupil_effects_by_sounds()


def plot_exmaple_pupil_psths():
    """
    Supplementary figure 2, panels A and B. Shows an example neuron response to
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
    Supplementary figure 2 panel C. Shows a scatter plot of  instances
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


def plot_sound_effect_on_pupil():
    fig = make_subplots(rows=1, cols=2, shared_xaxes='all', shared_yaxes='all',
                        subplot_titles=('prior sound', 'current sound'))

    val_col = 'value'
    for mm, sound_source in enumerate(['context_name', 'probe_name']):
        print(f'\n######### statisitic for {sound_source} '
              f'effect on pupil #########\n')
        indf = pupil_by_sound_DF.query(
            "part == 'probe'")

        _ = kruskal_with_posthoc(indf, group_col=sound_source, val_col=val_col)
        indf = indf.groupby(by=sound_source, observed=True).agg(
            stat=(val_col, np.mean), err=(val_col, sst.sem),
            count=(val_col, 'count')
        ).sort_values(
            'stat', ascending=True
        ).reset_index()

        print('\ndata summary stats\n', indf)

        x = indf[sound_source]
        y = indf['stat']
        yerr = indf['err']

        fig.add_trace(
            go.Scatter(
                x=y, y=x, mode='markers',
                marker=dict(color='black', size=4),
                error_x=dict(array=yerr, color='black',
                             thickness=1, width=5),
                showlegend=False
            ),
            col=mm + 1, row=1
        )

    h, w = 4, 3.5
    fig.update_layout(
        template='simple_white', height=96 * h, width=96 * w,
        margin=dict(l=10, r=10, t=20, b=10),
        font_family='arial',
        yaxis=dict(title=dict(text='sound')),
        xaxis=dict(title=dict(text='pupil size', font_color='black')),
        xaxis2=dict(title=dict(text='pupil size', font_color='black')),
        showlegend=False
    )

    fig.update_xaxes(title_font_size=10, title_standoff=0, tickfont_size=9)
    fig.update_yaxes(title_font_size=10, title_standoff=0, tickfont_size=9)

    return fig


def print_sound_metrics_correlation_to_pupil():
    """
    Prints regression values of the relationship between a sound
    metric and the average pupil size during and after the sound presentation.
    Prints regression for the 3 metrics used to describe the sounds:
    'temporal_stationarity', 'bandwidth', 'spectral_correlations'

    Returns: None

    """
    try:
        sound_metric_DF = pd.read_csv(
            root_path / 'data' / 'sound_quantifications' / 'metric_table.csv',
            index_col=0
        ).replace(
            {'name': short_name_map}
        )
    except:
        print("Sound metric  DF not found.\nEnsure to run the function "
              "'create_sound_analysis_table' in table_1.py scrip first")
        return None

    print("correlations of prior (context_name) and current (probe_name) "
          "sound metrics to pupil size:\n")

    for mm, sound_source in enumerate(['context_name', 'probe_name']):
        # metrics os sound as columns in sound_metric_DF
        print(f"{sound_source}")
        for metric in [
            'temporal_stationarity', 'bandwidth', 'spectral_correlations'
        ]:
            print(f"    {metric}")

            indf = pd.merge(
                left=pupil_by_sound_DF.query(
                    "part == 'probe'"
                ).loc[:, (sound_source, 'value')],
                right=sound_metric_DF.rename(
                    columns={'name': sound_source}).loc[:,
                      (sound_source, metric)],
                on=sound_source
            )

            x = indf[metric].values
            y = indf['value'].values

            reg = sst.linregress(x, y)
            print(f"    R={reg.rvalue:.3f}, p={reg.pvalue:.3f}")

    return None


def _plot_sound_metric_vs_pupil(sound_metric):
    """
    Scatter plots with regression line showing the relationship between a sound
    metric and the average pupil size during and after the sound presentation.
    The regressions are significant, but the R values are very small, so this
    shows that our metrics do not capture what the animal finds exciting about
    sounds

    Args:
        sound_metric: str, one of
        'temporal_stationarity', 'bandwidth', 'spectral_correlations'

    Returns:

    """
    try:
        sound_metric_DF = pd.read_csv(
            root_path / 'data' / 'sound_quantifications' / 'metric_table.csv',
            index_col=0
        ).replace(
            {'name': short_name_map}
        )
    except:
        print("Sound metric  DF not found.\nEnsure to run the function "
              "'create_sound_analysis_table' in table_1.py scrip first")
        return None

    fig = make_subplots(rows=2, cols=1, shared_xaxes='all', shared_yaxes='all',
                        subplot_titles=['prior sound', 'current sound'])

    for mm, sound_source in enumerate(['context_name', 'probe_name']):
        indf = pd.merge(
            left=pupil_by_sound_DF.query("part == 'probe'").loc[:,
                 (sound_source, 'value')],
            right=sound_metric_DF.rename(columns={'name': sound_source}).loc[:,
                  (sound_source, sound_metric)],
            on=sound_source
        )

        x = indf[sound_metric].values
        y = indf['value'].values

        reg = sst.linregress(x, y)

        print(reg)
        x_range = (x.max() - x.min()) * 0.01
        jitter = np.random.uniform(-x_range, x_range, x.size)

        fig.add_trace(
            go.Scattergl(
                x=x + jitter, y=y, mode='markers',
                marker=dict(color=add_opacity(Black, 0.3), size=2),
                showlegend=False
            ),
            col=1, row=mm + 1
        )

        # regression lines
        rx = np.array([x.min(), x.max()])
        ry = rx * reg.slope + reg.intercept
        fig.add_trace(
            go.Scatter(
                x=rx, y=ry, mode='lines',
                line=dict(color='green', width=2),
                name=f"r={reg.rvalue:.3f}, p={reg.pvalue:.3f}",
                showlegend=True
            ),
            col=1, row=mm + 1
        )

    h, w = 3, 3
    fig.update_layout(
        template='simple_white', height=96 * h, width=96 * w,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis2=dict(title=dict(text=sound_metric)),
        yaxis=dict(
            title=dict(text='pupil size (z-score)', font_color='black')),
        yaxis2=dict(
            title=dict(text='pupil size (z-score)', font_color='black')),
        showlegend=True,
        legend=dict(x=1, y=0.6, xanchor='right', yanchor='top', font_size=8,
                    bgcolor="rgba(0,0,0,0)", )
    )

    fig.update_xaxes(title_font_size=10, title_standoff=0, tickfont_size=9)
    fig.update_yaxes(title_font_size=10, title_standoff=0, tickfont_size=9)

    return fig
