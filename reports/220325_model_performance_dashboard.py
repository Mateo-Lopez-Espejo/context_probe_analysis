import pathlib as pl
from configparser import ConfigParser
from time import time

import joblib as jl
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback_context
from plotly.subplots import make_subplots

from src.models.modelnames import modelnames
from src.root_path import config_path
from src.utils.subsets import cellid_subset_02, cellid_A1_fit_set, cellid_PEG_fit_set
from src.visualization.palette import ColorList
from src.visualization.interactive import plot_raw_pair, plot_time_ser_quant, plot_strf, plot_pop_stategain, \
    plot_pop_modulation, plot_errors_over_time, plot_multiple_errors_over_time, plot_model_prediction_comparison

#### general configuration to import the right data and caches
config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 11000,
        'zscore': True,
        'stim_type': 'permutations'}

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220310_ctx_mod_metric_DF_tstat_cluster_mass_BS'
model_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220324_ctx_mod_metric_DF_pred'

# quick cache
dash_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220324_model_dashboad'

# TNC014a best example for the model fitting subset
start_prb = 8 - 1
start_ctxp = '00_08'
start_cellid = 'TNC014a-22-2'
batch = 326

# we dont need all models tryed so far
selected = ['STRF_long_relu', 'self_mod_relu', 'pop_mod_relu', 'self_lone_relu',
            'pop_lone_relu']  # all displayed modelnames (must contain STRF)
selected_mod = ['self_mod_relu', 'pop_mod_relu']  # subset of modelnames with with stategain
sel_real_vs_pred = ['STRF_long_relu', 'pop_mod_relu']
modelnames = {nickname: modelname for nickname, modelname in modelnames.items() if nickname in selected}

# truncate = '' # full time series metrics values
truncate = '_trunc1.5'  # truncated time series metrics
metrics = ['integral', 'mass_center', 'integral_trunc1.5', 'mass_center_trunc1.5', 'last_bin']


### load and preformat some of the main data

def filter_DF(DF):
    cellids = list(cellid_A1_fit_set.union(cellid_PEG_fit_set))
    # different filtering for actual and predicted data
    if {'mult_comp_corr', 'source', 'cluster_threshold'}.issubset(set(DF.columns)):
        # asumes real data
        filtered = DF.query(
            f"metric in {metrics} and mult_comp_corr == 'bf_cp' and source == 'real' and "
            "cluster_threshold == 0.05 and "
            f"id in {cellids} and value > 0"
        )
        df_indices = ['region', 'stim_count', 'context_pair', 'probe', 'id', 'site']
        is_real = True

    else:
        # asumes prediction
        filtered = DF.query(f"metric in {metrics} and "
                            f"id in {cellids} and "
                            f"nickname in {selected}"
                            )
        df_indices = ['region', 'stim_count', 'context_pair', 'probe', 'id', 'site', 'nickname', 'modelname']
        is_real = False

    pivoted = filtered.pivot_table(index=df_indices,
                                   columns=['metric'], values='value', aggfunc='first').reset_index()

    if is_real:
        # takes only significant data
        pivoted = pivoted.query('last_bin > 0')
        # adds a small amount of jitter to the last bin value to help visualization
        binsize = 1 / meta['raster_fs']
        jitter = (np.random.random(pivoted.shape[0]) * binsize * 0.8 - (binsize * 0.8 / 2)) * 1000  # in ms
        pivoted['last_bin'] = pivoted['last_bin'] + jitter

    return filtered, pivoted


recache = True
if dash_DF_file.exists() and not recache:
    print('found dash cache, loading ...')
    filtered, pivoted_full = jl.load(dash_DF_file)
else:
    if not dash_DF_file.parent.exists():
        dash_DF_file.parent.mkdir(parents=True, exist_ok=True)

    print('loading and formatting summary dataframe')
    tic = time()
    filtered, pivoted = filter_DF(jl.load(summary_DF_file))
    _, pivoted_pred = filter_DF(jl.load(model_DF_file))

    # keeps predicions common to significant real values
    pivoted_full = pivoted.merge(pivoted_pred, on=['site', 'region', 'id', 'context_pair', 'probe', 'stim_count'],
                                 suffixes=['_resp', '_pred'],
                                 validate='1:m')
    del (pivoted_pred, pivoted)  # no need for these big bois

    print(f'it took {time() - tic:.3f}s to load and format. cacheing')
    jl.dump([filtered, pivoted_full], dash_DF_file)

#### dashboard general layout

app = Dash(__name__)

# plots response metric space
# filters a single model to avoid repeated data points
toplot = pivoted_full.query(f"nickname == '{list(modelnames.keys())[0]}'")


dur_vs_amp = px.scatter(data_frame=pivoted_full, x="last_bin", y="integral_resp", color='id'
                        , hover_name='id', hover_data=['context_pair', 'probe'])

# # todo transform to use scattergl
# all_traces = list()
# for cc, (cellid) in enumerate(pivoted_full.id.unique()):
#     color = ColorList[cc%len(ColorList)]
#     df = pivoted_full.query(f"id == '{cellid}'")
#     trace = go.Scattergl(x=df['last_bin'], y=df['integral_resp'],
#                          mode='markers', marker_color=color,
#                          customdata=df.loc[:, ['context_pair', 'probe']],
#                          name=cellid,
#                          text=df['id'],
#                          hovertext=cellid)
# dur_vs_amp = go.Figure()
# dur_vs_amp.add_traces(all_traces)

nickname = 'pop_mod_relu'  # plot metric comparison between real and model data
raw_type = 'psth'  # real data display type
split_errors = False  # plot errors over time one model at a time


def callbacks_to_input(real_pic, amp_pic, dur_pic):
    """
    simple function to determine the last on click call back, and extract from it the necesary parameters for all
    the plotting functions
    """

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    # note the leading '' to catch when there has been no click at dashboard lauch
    which_click = {'': real_pic, 'dur_vs_amp': real_pic, 'real_vs_pred_integral': amp_pic,
                   'real_vs_pred_mass_center': dur_pic}
    if trigger_id not in which_click.keys():
        trigger_id = 'dur_vs_amp' # default for other callbacks not coming from the pi. this is kinda broken
    picked_eg = which_click[trigger_id]

    cellid = picked_eg['points'][0]['hovertext']
    contexts = [int(ss) for ss in picked_eg['points'][0]['customdata'][0].split('_')]
    probes = picked_eg['points'][0]['customdata'][1]

    return cellid, contexts, probes


def plot_real_vs_model(nicknames, parameter):
    # todo, make so all models can be together??
    toplot = pivoted_full.query(f"nickname in  {nicknames} and {parameter}_resp > 0")
    fig = px.scatter(data_frame=toplot, x=f'{parameter}_resp', y=f'{parameter}_pred', color='nickname',
                     hover_name='id', hover_data=['context_pair', 'probe'], trendline='ols')
    return fig


@app.callback(
    Output(component_id='sample_details', component_property='figure'),
    [Input(component_id='dur_vs_amp', component_property='clickData'),
     Input(component_id='real_vs_pred_integral', component_property='clickData'),
     Input(component_id='real_vs_pred_mass_center', component_property='clickData')]
)
def _plot_sample_details(real_pic, amp_pic, dur_pic):
    print('############################\nplotting sample details\n')
    cellid, contexts, probe = callbacks_to_input(real_pic, amp_pic, dur_pic)

    fig = make_subplots(1, 2, subplot_titles=(f'contexts: {contexts}; probe: {probe}', 'significant regions p<0.05'))
    psth = plot_raw_pair(cellid, contexts, probe, type=raw_type)
    quant_diff = plot_time_ser_quant(cellid, contexts, probe,
                                     multiple_comparisons_axis=[1, 2], consecutive=0, cluster_threshold=0.05,
                                     fn_name='big_shuff', meta=meta)

    fig.add_traces(psth['data'], rows=[1] * len(psth['data']), cols=[1] * len(psth['data']))
    fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1, row=1, col=1)
    fig.add_traces(quant_diff['data'], rows=[1] * len(quant_diff['data']), cols=[2] * len(quant_diff['data']))

    fig.update_layout(title_text=f'{cellid}')

    return fig


@app.callback(
    Output(component_id='model_predictions', component_property='figure'),
    [Input(component_id='dur_vs_amp', component_property='clickData'),
     Input(component_id='real_vs_pred_integral', component_property='clickData'),
     Input(component_id='real_vs_pred_mass_center', component_property='clickData')]
)
def _plot_multiple_predictions(real_pic, amp_pic, dur_pic):
    tic = time()
    print('############################\nplotting \n')
    cellid, contexts, probe = callbacks_to_input(real_pic, amp_pic, dur_pic)

    fig = make_subplots(1, len(modelnames), horizontal_spacing=0.05, subplot_titles=list(modelnames.keys()),
                        shared_xaxes=True, shared_yaxes=True)

    for midx, (name, modelname) in enumerate(modelnames.items()):
        psth = plot_raw_pair(cellid, contexts, probe, type='psth', modelname=modelname, batch=batch)
        fig.add_traces(psth['data'], rows=[1] * len(psth['data']), cols=[midx + 1] * len(psth['data']))
        fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1, row=1, col=midx + 1)

    fig.update_layout(showlegend=False)
    print(f'\nmultiple predictions done, took:{time() - tic:.2f}s\n############################')
    return fig


# @app.callback(
#     Output(component_id='model_comparisons', component_property='figure'),
#     [Input(component_id='dur_vs_amp', component_property='clickData'),
#      Input(component_id='real_vs_pred_integral', component_property='clickData'),
#      Input(component_id='real_vs_pred_mass_center', component_property='clickData')]
# )
# def _plot_model_comparisons(real_pic, amp_pic, dur_pic):
#     tic = time()
#     print('############################\nplotting model pred comparisons\n')
#     cellid, contexts, probe = callbacks_to_input(real_pic, amp_pic, dur_pic)
#
#     independent_models = [modelnames['STRF_long_relu'], modelnames['pop_lone_relu']]
#     dependent_model = modelnames['pop_mod_relu']
#     fig = plot_model_prediction_comparison(cellid, batch, independent_models, dependent_model, contexts, probe)
#     print(f'\nmodel pred coparisosn done, took: {time() - tic:.2f}s\n############################')
#     return fig


@app.callback(
    Output(component_id='model_errors', component_property='figure'),
    [Input(component_id='dur_vs_amp', component_property='clickData'),
     Input(component_id='real_vs_pred_integral', component_property='clickData'),
     Input(component_id='real_vs_pred_mass_center', component_property='clickData'),
     Input(component_id='error_style', component_property='value')]
)
def _plot_multiple_errors(real_pic, amp_pic, dur_pic, error_style):
    tic = time()
    print('############################\nplotting overlayed errors over time\n')
    cellid, contexts, probe = callbacks_to_input(real_pic, amp_pic, dur_pic)
    if split_errors:
        fig = make_subplots(1, len(modelnames), horizontal_spacing=0.05, subplot_titles=list(modelnames.keys()),
                            shared_xaxes=True, shared_yaxes=True)

        for midx, (name, modelname) in enumerate(modelnames.items()):
            errors = plot_errors_over_time(cellid, modelname, batch, contexts, probe, part='probe', grand_mean=error_style)
            fig.add_traces(errors['data'], rows=[1] * len(errors['data']), cols=[midx + 1] * len(errors['data']))
            fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1, row=1, col=midx + 1)

        fig.update_layout(showlegend=False)

    else:
        fig = plot_multiple_errors_over_time(cellid, list(modelnames.values()), batch, contexts, probe, part='probe',
                                             style=error_style, floor=modelnames['STRF_long_relu'])

    print(f'\noverlayed errors done, took: {time() - tic:.2f}s\n############################')
    return fig


@app.callback(
    Output(component_id='model_strfs', component_property='figure'),
    [Input(component_id='dur_vs_amp', component_property='clickData'),
     Input(component_id='real_vs_pred_integral', component_property='clickData'),
     Input(component_id='real_vs_pred_mass_center', component_property='clickData')]
)
def _plot_multiple_strf(real_pic, amp_pic, dur_pic):
    tic = time()
    print('############################\nplotting multiple strf\n')
    cellid, _, _ = callbacks_to_input(real_pic, amp_pic, dur_pic)

    fig = make_subplots(1, len(modelnames), horizontal_spacing=0.05, subplot_titles=list(modelnames.keys()), )

    for midx, (name, modelname) in enumerate(modelnames.items()):
        strf = plot_strf(cellid, modelname, batch)
        fig.add_traces(strf['data'], rows=[1] * len(strf['data']), cols=[midx + 1] * len(strf['data']))

    # make common colorbar
    fig.update_layout(coloraxis=dict(colorscale='inferno',
                                     colorbar=dict(
                                         thickness=10, len=0.6,
                                         title_text='weight',
                                         title_side='right',
                                         tickangle=-90,
                                         xanchor='left', x=1)
                                     ),
                      showlegend=False
                      )

    print(f'\nmultiple STRFs done, took: {time() - tic:.2f}s\n############################')
    return fig


@app.callback(
    Output(component_id='model_stategains', component_property='figure'),
    [Input(component_id='dur_vs_amp', component_property='clickData'),
     Input(component_id='real_vs_pred_integral', component_property='clickData'),
     Input(component_id='real_vs_pred_mass_center', component_property='clickData')]
)
def _plot_multiple_stategains(real_pic, amp_pic, dur_pic):
    tic = time()
    print('############################\nplotting multiple state gains\n')
    cellid, contexts, probe = callbacks_to_input(real_pic, amp_pic, dur_pic)

    fig = make_subplots(2, len(modelnames), row_width=[0.05, 0.95], vertical_spacing=0.01, horizontal_spacing=0.05,
                        shared_yaxes=True,
                        subplot_titles=list(modelnames.keys()) + ([''] * len(modelnames))  # title only on top row
                        )

    for mm, (nickname, modelname) in enumerate(modelnames.items()):
        if nickname not in selected_mod:
            # skips those models without stategain
            continue
        mod_plot = plot_pop_modulation(cellid, modelnames[nickname], batch, contexts, probe)
        weight_plot = plot_pop_stategain(cellid, modelnames[nickname], batch, orientation='h')

        fig.add_traces(mod_plot['data'], rows=[1] * len(mod_plot['data']), cols=[mm + 1] * len(mod_plot['data']))
        fig.add_traces(weight_plot['data'], rows=[2] * len(weight_plot['data']),
                       cols=[mm + 1] * len(weight_plot['data']))

        _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1,
                          row=1, col=mm + 1)

    # make common colorbar
    fig.update_layout(coloraxis=dict(colorscale='inferno',
                                     colorbar=dict(
                                         thickness=10, len=0.6,
                                         title_text='weight',
                                         title_side='right',
                                         tickangle=-90,
                                         xanchor='left',
                                         orientation='v')
                                     )
                      )

    print(f'\nmultiple state gains done, took: {time() - tic:.2f}s\n############################')
    return fig


app.layout = html.Div([
    dcc.Graph(
        id='dur_vs_amp',
        figure=dur_vs_amp,
        clickData={'points': [{'hovertext': start_cellid,
                               'customdata': [start_ctxp, start_prb]}]}
    ),
    html.Div([dcc.Graph(id=f'real_vs_pred_{parameter}',
                        figure=plot_real_vs_model(sel_real_vs_pred, f'{parameter}{truncate}'),
                        clickData={'points': [{'hovertext': start_cellid,
                                               'customdata': [start_ctxp, start_prb]}]}
                        )
              for parameter in ['integral', 'mass_center']

              ], style={'display': 'flex', 'flex-direction': 'row'}
             ),
    dcc.Graph(
        id='sample_details'
    ),
    dcc.Graph(
        id='model_predictions'
    ),
    # dcc.Graph(
    #     id='model_comparisons'
    # ),
    html.Div(children=[
        dcc.RadioItems(id='error_style', options=['mean', 'instance','PCA'], value='mean'),
        dcc.Graph(id='model_errors')
    ]),
    dcc.Graph(
        id='model_strfs'
    ),
    dcc.Graph(
        id='model_stategains'
    ),
])
if __name__ == '__main__':
    print("inside __name__ == '__main__'")
    app.run_server(debug=True)
