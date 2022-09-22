import pathlib as pl
from configparser import ConfigParser
from time import time

import joblib as jl
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback_context
from plotly.subplots import make_subplots

from src.models.modelnames import modelnames
from src.root_path import config_path
from src.utils.subsets import batch_map
from src.utils.subsets import cellid_A1_fit_set, cellid_PEG_fit_set
from src.visualization.interactive import plot_raw_pair, plot_time_ser_quant, plot_strf, plot_pop_stategain, \
    plot_pop_modulation, plot_errors_over_time, plot_multiple_errors_over_time

#### general configuration to import the right data and caches
config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,  # originally 30
        'montecarlo': 11000,
        'zscore': True,
        'stim_type': 'permutations'}

# source raw DFs
summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220520_minimal_DF'
model_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220412_resp_pred_metrics_by_chunks'

# quick cache
dash_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220324_model_dashboad'
recache_DF = True

# TNC014a best example for the model fitting subset
start_prb = 8 - 1
start_ctxp = '00_08'
start_cellid = 'TNC014a-22-2'

selected = ['matchl_STRF', 'matchl_self', 'matchl_pop',
            'matchl_full']  # all displayed modelnames (must contain STRF) # all displayed modelnames (must contain STRF)
selected_mod = selected
modelnames = {sel: modelnames[sel] for sel in selected}

# metrics = ['integral', 'mass_center', 'integral_trunc1.5', 'mass_center_trunc1.5', 'last_bin']
metrics = ['integral', 'mass_center', 'last_bin']

cellids = list(cellid_A1_fit_set.union(cellid_PEG_fit_set))

cellids = ["TNC014a-22-2"]


### load and preformat some of the main data

def get_formated_DF():
    # real values, defines filtering
    DF_mass = jl.load(summary_DF_file)
    filtered = DF_mass.query(f"metric in {metrics} and mult_comp_corr == 'bf_cp' and source == 'real' and "
                             f"cluster_threshold == 0.05 and "
                             f"id in {cellids} and value > 0")

    pivoted_clust_mass = filtered.pivot_table(
        index=['region', 'stim_count', 'context_pair', 'probe', 'id', 'site'],
        columns=['metric'], values='value', aggfunc='first', fill_value=0, observed=True,
    ).reset_index().query('last_bin > 0')

    # adds a small amount of jitter to the last bin value to help visualization
    binsize = 1 / meta['raster_fs']
    jitter = (np.random.random(pivoted_clust_mass.shape[0]) * binsize * 0.8 - (binsize * 0.8 / 2)) * 1000  # in ms
    pivoted_clust_mass['last_bin'] = pivoted_clust_mass['last_bin'] + jitter

    good_instances = pivoted_clust_mass.loc[:, ['id', 'context_pair', 'probe']].drop_duplicates()

    # cleanup big unused dataframes
    del (DF_mass, filtered)

    # model related metrics
    DF = jl.load(model_DF_file)
    pred_filtered = DF.query(f"metric in {metrics} and "
                             f"id in {cellids} and "
                             f"nickname in {selected} and time_bin == 'full'"
                             )

    resp_filtered = DF.query(f"metric in {metrics} and "
                             f"id in {cellids} and "
                             f"nickname == 'response' and time_bin == 'full'"
                             )

    # fold the response, common to all model predicitons, and turns it along model prediction into columns
    pivoted_models = pd.merge(
        left=resp_filtered, right=pred_filtered,
        on=['id', 'site', 'region', 'context_pair', 'probe', 'metric', 'stim_count'],
        validate='1:m'
    ).pivot_table(
        index=['id', 'site', 'region', 'context_pair', 'probe', 'stim_count', 'value_x', 'nickname_y'],
        columns=['metric'], values='value_y', observed=True,
    ).reset_index().rename(columns={'value_x': 'response', 'nickname_y': 'nickname'})

    # filter by the subset of instanceses with significant contextual modulateion as measured by cluster-mass
    pivoted_models = pd.merge(left=pivoted_models, right=good_instances, on=['id', 'context_pair', 'probe'])

    return pivoted_clust_mass, pivoted_models


if dash_DF_file.exists() and not recache_DF:
    print('found dash cache, loading ...')
    pivoted_clust_mass, pivoted_models = jl.load(dash_DF_file)
else:
    if not dash_DF_file.parent.exists():
        dash_DF_file.parent.mkdir(parents=True, exist_ok=True)

    print('loading and formatting summary dataframe')

    tic = time()
    pivoted_clust_mass, pivoted_models = get_formated_DF()

    print(f'it took {time() - tic:.3f}s to load and format. cacheing...')

    jl.dump([pivoted_clust_mass, pivoted_models], dash_DF_file)
print('done')

# load dataframe for all model performances

#### dashboard general layout


app = Dash(__name__)

# plots response metric space per site

toplot = pivoted_clust_mass.groupby(['site', 'region'], observed=True).agg('mean').reset_index()

dur_vs_amp = px.scatter(data_frame=toplot, x="last_bin", y="integral", color='region',
                        hover_name='site')

nickname = 'pop_mod_relu'  # plot metric comparison between real and model data
raw_type = 'psth'  # real data display type
split_errors = False  # plot errors over time one model at a time


def callbacks_to_input(real_pic, mod_pic):
    """
    simple function to determine the last on click call back, and extract from it the necesary parameters for all
    the plotting functions
    """

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    # note the leading '' to catch when there has been no click at dashboard lauch
    which_click = {'': real_pic, 'dur_vs_amp': real_pic, 'real_vs_pred': mod_pic}
    if trigger_id not in which_click.keys():
        trigger_id = 'dur_vs_amp'  # default for other callbacks not coming from the pi. this is kinda broken
    picked_eg = which_click[trigger_id]

    cellid = picked_eg['points'][0]['hovertext']
    contexts = [int(ss) for ss in picked_eg['points'][0]['customdata'][0].split('_')]
    probes = picked_eg['points'][0]['customdata'][1]

    return cellid, contexts, probes


@app.callback(
    Output(component_id='cell_scat', component_property='figure'),
    Input(component_id='dur_vs_amp', component_property='clickData'),
    prevent_initial_call=True
)
def plot_cell_scatter(real_pic):
    tic = time()
    print('############################\nplotting cell scatter\n')
    print(real_pic)
    siteid = real_pic['points'][0]['hovertext']

    toplot = pivoted_clust_mass.query(f"site == '{siteid}'")

    fig = px.scatter(data_frame=toplot, x="last_bin", y="integral", color='id',
                     hover_name='id', hover_data=['context_pair', 'probe'])

    print(f'\ncell scatter done, took:{time() - tic:.2f}s\n############################')
    return fig


@app.callback(
    Output(component_id='real_vs_pred', component_property='figure'),
    [Input(component_id='dur_vs_amp', component_property='clickData'),
     Input(component_id='metric', component_property='value')],
    prevent_initial_call=True
)
def plot_real_vs_model(real_pic, metric):
    siteid = real_pic['points'][0]['hovertext']

    # toplot = pivoted_full.query(f"site == '{siteid}' and {metric}_resp > 0")
    toplot = pivoted_models.query(f"site == '{siteid}' and {metric} > 0")
    fig = px.scatter(data_frame=toplot, x='response', y=metric, color='nickname',
                     hover_name='id', hover_data=['context_pair', 'probe'], trendline='ols')
    return fig


@app.callback(
    Output(component_id='sample_details', component_property='figure'),
    [Input(component_id='cell_scat', component_property='clickData'),
     Input(component_id='real_vs_pred', component_property='clickData')],
    prevent_initial_call=True
)
def _plot_sample_details(real_pic, mod_pic):
    tic = time()
    print('############################\nplotting sample details\n')
    cellid, contexts, probe = callbacks_to_input(real_pic, mod_pic)

    fig = make_subplots(1, 2, subplot_titles=(f'contexts: {contexts}; probe: {probe}', 'significant regions p<0.05'))
    psth = plot_raw_pair(cellid, contexts, probe, mode=raw_type)
    quant_diff = plot_time_ser_quant(cellid, contexts, probe,
                                     multiple_comparisons_axis=[1, 2], consecutive=0, cluster_threshold=0.05,
                                     meta=meta)

    fig.add_traces(psth['data'], rows=[1] * len(psth['data']), cols=[1] * len(psth['data']))
    fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1, row=1, col=1)
    fig.add_traces(quant_diff['data'], rows=[1] * len(quant_diff['data']), cols=[2] * len(quant_diff['data']))

    fig.update_layout(title_text=f'{cellid}')
    print(f'\nsample details done, took:{time() - tic:.2f}s\n############################')
    return fig


@app.callback(
    Output(component_id='model_predictions', component_property='figure'),
    [Input(component_id='cell_scat', component_property='clickData'),
     Input(component_id='real_vs_pred', component_property='clickData')],
    prevent_initial_call=True
)
def _plot_multiple_predictions(real_pic, mod_pic):
    tic = time()
    print('############################\nplotting multiple prediction\n')
    cellid, contexts, probe = callbacks_to_input(real_pic, mod_pic)

    fig = make_subplots(1, len(modelnames), horizontal_spacing=0.05, subplot_titles=list(modelnames.keys()),
                        shared_xaxes=True, shared_yaxes=True)

    for midx, (name, modelname) in enumerate(modelnames.items()):
        psth = plot_raw_pair(cellid, contexts, probe, mode='psth', modelname=modelname, batch=batch_map[cellid])
        fig.add_traces(psth['data'], rows=[1] * len(psth['data']), cols=[midx + 1] * len(psth['data']))
        fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1, row=1, col=midx + 1)

    fig.update_layout(showlegend=False)
    print(f'\nmultiple predictions done, took:{time() - tic:.2f}s\n############################')
    return fig


@app.callback(
    Output(component_id='model_errors', component_property='figure'),
    [Input(component_id='cell_scat', component_property='clickData'),
     Input(component_id='real_vs_pred', component_property='clickData'),
     Input(component_id='error_style', component_property='value')],
    prevent_initial_call=True
)
def _plot_multiple_errors(real_pic, mod_pic, error_style):
    tic = time()
    print('############################\nplotting overlayed errors over time\n')
    cellid, contexts, probe = callbacks_to_input(real_pic, mod_pic)
    if split_errors:
        fig = make_subplots(1, len(modelnames), horizontal_spacing=0.05, subplot_titles=list(modelnames.keys()),
                            shared_xaxes=True, shared_yaxes=True)

        for midx, (name, modelname) in enumerate(modelnames.items()):
            errors = plot_errors_over_time(cellid, modelname, batch_map[cellid], contexts, probe, part='probe',
                                           grand_mean=error_style)
            fig.add_traces(errors['data'], rows=[1] * len(errors['data']), cols=[midx + 1] * len(errors['data']))
            fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1, row=1, col=midx + 1)

        fig.update_layout(showlegend=False)

    else:
        fig = plot_multiple_errors_over_time(cellid, list(modelnames.values()), batch_map[cellid], contexts, probe,
                                             part='probe',
                                             style=error_style, floor=modelnames['match_STRF'])

    print(f'\noverlayed errors done, took: {time() - tic:.2f}s\n############################')
    return fig


@app.callback(
    Output(component_id='model_strfs', component_property='figure'),
    [Input(component_id='cell_scat', component_property='clickData'),
     Input(component_id='real_vs_pred', component_property='clickData')],
    prevent_initial_call=True
)
def _plot_multiple_strf(real_pic, mode_pic):
    tic = time()
    print('############################\nplotting multiple strf\n')
    cellid, _, _ = callbacks_to_input(real_pic, mode_pic)

    fig = make_subplots(1, len(modelnames), horizontal_spacing=0.05, subplot_titles=list(modelnames.keys()), )

    for midx, (name, modelname) in enumerate(modelnames.items()):
        strf = plot_strf(cellid, modelname, batch_map[cellid])
        fig.add_traces(strf['data'], rows=[1] * len(strf['data']), cols=[midx + 1] * len(strf['data']))

    # make common colorbar
    fig.update_layout(coloraxis=dict(colorscale='BrBG',
                                     cmid=0,
                                     colorbar=dict(
                                         thickness=10, len=0.6,
                                         title_text='weight',
                                         title_side='right',
                                         tickangle=-45,
                                         xanchor='left', x=1)
                                     ),
                      showlegend=False
                      )

    print(f'\nmultiple STRFs done, took: {time() - tic:.2f}s\n############################')
    return fig


@app.callback(
    Output(component_id='model_stategains', component_property='figure'),
    [Input(component_id='cell_scat', component_property='clickData'),
     Input(component_id='real_vs_pred', component_property='clickData')],
    prevent_initial_call=True
)
def _plot_multiple_stategains(real_pic, mod_pic):
    tic = time()
    print('############################\nplotting multiple state gains\n')
    cellid, contexts, probe = callbacks_to_input(real_pic, mod_pic)

    fig = make_subplots(2, len(modelnames), row_width=[0.1, 0.9], vertical_spacing=0.01, horizontal_spacing=0.05,
                        shared_yaxes=True,
                        subplot_titles=list(modelnames.keys()) + ([''] * len(modelnames))  # title only on top row
                        )

    for mm, (nickname, modelname) in enumerate(modelnames.items()):
        if nickname not in selected_mod:
            # skips those models without stategain
            continue
        mod_plot = plot_pop_modulation(cellid, modelnames[nickname], batch_map[cellid], contexts, probe)
        weight_plot = plot_pop_stategain(cellid, modelnames[nickname], batch_map[cellid], orientation='h')

        fig.add_traces(mod_plot['data'], rows=[1] * len(mod_plot['data']), cols=[mm + 1] * len(mod_plot['data']))
        fig.add_traces(weight_plot['data'], rows=[2] * len(weight_plot['data']),
                       cols=[mm + 1] * len(weight_plot['data']))

        _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1,
                          row=1, col=mm + 1)

    # make common colorbar
    fig.update_layout(coloraxis=dict(colorscale='BrBG',
                                     cmid=0,
                                     colorbar=dict(
                                         thickness=10, len=0.6,
                                         title_text='weight',
                                         title_side='right',
                                         tickangle=-45,
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
    dcc.Graph(
        id='cell_scat',
    ),
    html.Div(children=[
        dcc.RadioItems(id='metric', options=['integral', 'mass_center'], value='integral'),
        dcc.Graph(id='real_vs_pred')
    ]),
    dcc.Graph(
        id='sample_details'
    ),
    dcc.Graph(
        id='model_predictions'
    ),
    html.Div(children=[
        dcc.RadioItems(id='error_style', options=['mean', 'instance', 'PCA'], value='mean'),
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
