import pathlib as pl
from configparser import ConfigParser

import joblib as jl
import numpy as np
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output

from src.root_path import config_path
from src.visualization.palette import *

#### general configuration to import the right data and caches
config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None,
        'stim_type': 'permutations',
        'alpha': 0.05}
# todo, if batch analysis rerun, use the anotated line instead
# summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'211221_cxt_metrics_summary_DF_alpha_{meta}'
summary_DF_file = pl.Path(config['paths']['analysis_cache']) / '211221_cxt_metrics_summary_DF_alpha_0.05'

### same example cell as in figure 1 ###
prb_idx = 3 - 1  # selected probe. the -1 is to acount for 0 not being used
ctx_pair = [0, 1]  # pair of contexts to compare and exemplify d'
cellid = 'ARM021b-36-8'

### load and preformat some of the main data
DF = jl.load(summary_DF_file)


def format_dataframe(DF):
    ff_analylis = DF.analysis.isin(['SC', 'fdPCA'])
    ff_corr = DF.mult_comp_corr == 'consecutive_3'

    good_cols = ['analysis', 'mult_comp_corr', 'region', 'siteid', 'cellid', 'context_pair',
                 'probe', 'metric', 'value']
    filtered = DF.loc[ff_analylis & ff_corr, good_cols]

    filtered['probe'] = [int(p) for p in filtered['probe']]
    filtered['context_pair'] = [f"{int(cp.split('_')[0]):02d}_{int(cp.split('_')[1]):02d}"
                                for cp in filtered['context_pair']]

    # rename metrics and analysis for ease of ploting
    filtered['metric'] = filtered['metric'].replace({'significant_abs_mass_center': 'center of mass (ms)',
                                                     'significant_abs_mean': "mean d'",
                                                     'significant_abs_sum': "integral (d'*ms)"})
    filtered['analysis'] = filtered['analysis'].replace({'SC': 'single cell',
                                                         'fdPCA': 'population',
                                                         'pdPCA': 'probewise pop',
                                                         'LDA': 'pop ceiling'})

    filtered['id'] = filtered['cellid'].fillna(value=filtered['siteid'])
    filtered = filtered.drop(columns=['cellid', 'siteid'])

    filtered['value'] = filtered['value'].fillna(value=0)

    # permutation related preprocesing.
    # creates a new column relating probe with  context pairs
    ctx = np.asarray([row.split('_') for row in filtered.context_pair], dtype=int)
    prb = np.asarray(filtered.probe, dtype=int)

    silence = ctx == 0
    same = ctx == prb[:, None]
    different = np.logical_and(~silence, ~same)

    name_arr = np.full_like(ctx, np.nan, dtype=object)
    name_arr[silence] = 'silence'
    name_arr[same] = 'same'
    name_arr[different] = 'diff'
    comp_name_arr = np.apply_along_axis('_'.join, 1, name_arr)

    # swaps clasification names to not have repetitions i.e. diff_same == same_diff
    comp_name_arr[np.where(comp_name_arr == 'same_silence')] = 'silence_same'
    comp_name_arr[np.where(comp_name_arr == 'diff_silence')] = 'silence_diff'
    comp_name_arr[np.where(comp_name_arr == 'diff_same')] = 'same_diff'
    comp_name_arr[np.where(comp_name_arr == 'same_silence')] = 'silence_same'

    filtered['trans_pair'] = comp_name_arr
    return filtered


DF_long = format_dataframe(DF)

### readies the first figure
# for the first figure take the cell mean value to reduce the clutter and call one cell at a time

# ff_analysis = DF_long.analysis == 'single cell'
# filtered = DF_long.loc[ff_analysis, :].groupby(['id', 'region', 'metric']).agg({'value':'mean'}).reset_index()
pivoted = DF_long.pivot_table(index=['analysis', 'id', 'region'], columns=['metric'], values='value', aggfunc='mean')
pivoted = pivoted.loc['single cell']
dur_vs_amp_df = pivoted.loc[(pivoted['center of mass (ms)'] > 0), :].reset_index()


### simplified version of the tile figure ###
# prefilters all the data to be ploted (multiple neurons) to ensure shared colormaps
# for the fill values

def _plot_neuron_tiling(picked_neuron):
    # turns long format data into an array with dimension Probe * context_pair
    to_pivot = DF_long.loc[DF_long['id'] == picked_neuron, :]
    val_df = to_pivot.pivot_table(index=['metric', 'probe'], columns=['context_pair'], values='value')

    cscales = {"integral (d'*ms)": pc.make_colorscale(['#000000', Green]),
               "center of mass (ms)": pc.make_colorscale(['#000000', Purple])}
    max_vals = dict()
    # normalizes,saves max values and get colors for each metric
    color_df = val_df.copy()
    for metric in color_df.index.levels[0]:
        max_vals[metric] = val_df.loc[metric].values.max()
        colors = pc.sample_colorscale(cscales[metric],
                                      (val_df.loc[metric] / max_vals[metric]).values.flatten())
        color_df.loc[metric] = np.asarray(colors).reshape(color_df.loc[metric].shape)

    # general shapes of the upper and lower triangles to be passed to Scatter x and y
    xu, yu = np.array([0, 0, 1, 0]), np.array([0, 1, 1, 0])
    xl, yl = np.array([0, 1, 1, 0]), np.array([0, 0, 1, 0])

    amp_color = color_df.loc[("integral (d'*ms)"), :].values
    dur_color = color_df.loc[('center of mass (ms)'), :].values

    amplitudes = val_df.loc[("integral (d'*ms)"), :]
    durations = val_df.loc[('center of mass (ms)'), :]

    fig = go.Figure()

    for nn, (p, c) in enumerate(np.ndindex(amp_color.shape)):
        # note the use of transparent markers to define the colorbars internally
        # amplitud uppe half
        _ = fig.add_scatter(x=xu + c, y=yu + p, mode='lines+markers',
                            line_width=1, line_color='#222222',
                            fill='toself', fillcolor=amp_color[p, c],
                            marker=dict(color=(amplitudes.values[p, c],) * len(xu),
                                        coloraxis='coloraxis',
                                        opacity=0,
                                        cmin=0, cmax=max_vals["integral (d'*ms)"],
                                        ),
                            showlegend=False
                            )

        # duration lower half
        _ = fig.add_scatter(x=xl + c, y=yl + p, mode='lines+markers',
                            line_width=1, line_color='#222222',
                            fill='toself', fillcolor=dur_color[p, c],
                            marker=dict(color=(durations.values[p, c],) * len(xl),
                                        coloraxis='coloraxis2',
                                        opacity=0,
                                        cmin=0, cmax=max_vals["center of mass (ms)"],
                                        ),
                            showlegend=False
                            )

    # strip left zero padding for better display
    ticktexts = [f"{int(pp.split('_')[0])}_{int(pp.split('_')[1])}"
                 for pp in amplitudes.columns.to_list()]
    _ = fig.update_xaxes(dict(scaleanchor=f'y',
                              constrain='domain',
                              range=[0, amplitudes.columns.size], fixedrange=True,
                              title_text='context pairs',
                              tickmode='array',
                              tickvals=np.arange(amplitudes.columns.size) + 0.5,
                              ticktext=ticktexts))

    _ = fig.update_yaxes(dict(title=dict(text=f'{picked_neuron}<br>probes'),
                              constrain='domain',
                              range=[0, amplitudes.index.size], fixedrange=True,
                              tickmode='array',
                              tickvals=np.arange(amplitudes.index.size) + 0.5,
                              ticktext=amplitudes.index.to_list()))

    # set the positions of the colorbars
    fig.update_layout(coloraxis=dict(colorscale=cscales["integral (d'*ms)"],
                                     colorbar=dict(
                                         thickness=10, len=0.6,
                                         title_text="amplitude (d'*ms)",
                                         title_side='right',
                                         tickangle=-90,
                                         xanchor='left', x=1)
                                     ),
                      coloraxis2=dict(colorscale=cscales["center of mass (ms)"],
                                      colorbar=dict(
                                          thickness=10, len=0.6,
                                          title_text="duration (ms)",
                                          title_side='right',
                                          tickangle=-90,
                                          xanchor='left', x=1.1)
                                      )
                      )

    return fig


#### dashboard general layout

app = Dash(__name__)

dur_vs_amp = px.scatter(data_frame=dur_vs_amp_df, x="center of mass (ms)", y="integral (d'*ms)", color='region',
                        color_discrete_sequence=[Blue, Orange], hover_name='id')

app.layout = html.Div([
    dcc.Graph(
        id='dur_vs_amp',
        figure=dur_vs_amp,
        clickData={'points': [{'hovertext': cellid}]}
    ),
    dcc.Graph(
        id='space_tile'
    )
])


@app.callback(
    Output(component_id='space_tile', component_property='figure'),
    Input(component_id='dur_vs_amp', component_property='clickData')
)
def plot_neuron_tiling(picked_neuron):
    picked_neuron = picked_neuron['points'][0]['hovertext']
    return _plot_neuron_tiling(picked_neuron)


if __name__ == '__main__':
    app.run_server(debug=True)

