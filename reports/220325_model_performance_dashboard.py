import pathlib as pl
from configparser import ConfigParser
from time import time

import joblib as jl
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback_context
from plotly.subplots import make_subplots

from src.root_path import config_path
from src.visualization.interactive import plot_raw_pair, plot_time_ser_quant, plot_strf, plot_pop_stategain, plot_pop_modulation

#### general configuration to import the right data and caches
config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'stim_type': 'permutations'}

meta_BS = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 11000,
        'zscore': True,
        'stim_type': 'permutations'}

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220310_ctx_mod_metric_DF_tstat_cluster_mass_BS'
model_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220324_ctx_mod_metric_DF_pred'

# quick cache
dash_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220324_model_dashboad'


### same example cell as in figure 1 ###
start_prb = 9 - 1  # selected probe. the -1 is to acount for 0 not being used
start_ctxp = '00_01'  # pair of contexts to compare and exemplify d'
start_cellid = 'ARM021b-36-8'

# TNC014a best example for the model fitting subset
start_prb = 8 - 1
start_ctxp = '00_08'
start_cellid = 'TNC014a-22-2'


batch = 326

### load and preformat some of the main data

def filter_DF(DF):
    sites = ['TNC014a', 'TNC023a', 'TNC018a', 'TNC015a', 'TNC017a', 'TNC016a', 'TNC009a', 'TNC024a', 'TNC010a'] # all NTI-CPN10
    sites = ['TNC014a'] #good example for fits

    # different filtering for actual and predicted data
    if {'mult_comp_corr', 'source', 'cluster_threshold'}.issubset(set(DF.columns)):
        # asumes real data
        filtered = DF.query("metric in ['integral', 'mass_center', 'last_bin'] and mult_comp_corr == 'bf_cp' and source == 'real' and "
                            "cluster_threshold == 0.05 and "
                            f"site in {sites}")
        df_indices = ['source', 'mult_comp_corr', 'cluster_threshold', 'region',
                      'stim_count', 'context_pair', 'probe', 'id', 'site']
        is_real = True

    else:
        # asumes prediction
        filtered = DF.query(f"metric in ['integral', 'mass_center'] and site in {sites}")
        df_indices = ['region', 'stim_count', 'context_pair', 'probe', 'id', 'site']
        is_real = False


    pivoted = filtered.pivot_table(index=df_indices,
                                   columns=['metric'], values='value', aggfunc='first').reset_index()

    if is_real:
        # takes only significant data
        pivoted = pivoted.query('last_bin > 0')
        # adds a small amount of jitter to the last bin value to help visualization
        binsize = 1/meta['raster_fs']
        jitter = (np.random.random(pivoted.shape[0]) * binsize * 0.8 - (binsize*0.8/2)) * 1000 # in ms
        pivoted['last_bin'] = pivoted['last_bin'] + jitter

    return filtered, pivoted

recache = False
if dash_DF_file.exists() and not recache:
    print('found dash cache, loading ...')
    filtered, pivoted_full = jl.load(dash_DF_file)
else:
    if not dash_DF_file.parent.exists():
        dash_DF_file.parent.mkdir(parents=True,exist_ok=True)

    print('loading and formatting summary dataframe')
    tic = time()
    filtered, pivoted = filter_DF(jl.load(summary_DF_file))
    _, pivoted_pred = filter_DF(jl.load(model_DF_file))

    # keeps predicions common to significant real values
    pivoted_full = pivoted.merge(pivoted_pred.loc[:, ['id', 'context_pair', 'probe', 'integral', 'mass_center']],
                        on=['id', 'context_pair', 'probe'], suffixes=['_resp', '_pred'], validate='1:m')
    del (pivoted_pred, pivoted) # no need for these big bois

    print(f'it took {time() - tic:.3f}s to load and format. cacheing')
    jl.dump([filtered, pivoted_full], dash_DF_file)


#### dashboard general layout

app = Dash(__name__)

# plots response metric space
dur_vs_amp = px.scatter(data_frame=pivoted_full, x="last_bin", y="integral_resp", color='region'
                        , hover_name='id', hover_data=['context_pair', 'probe'],
                        marginal_x='histogram', marginal_y='histogram')

# plot metric comparison between real and model data

def plot_real_vs_model(modelname, parameter):
    #todo, make so all models can be together??
    print(f'modelname specified but not yet suported, displaying results for population_modulation model')

    fig = px.scatter(data_frame=pivoted_full, x=f'{parameter}_resp', y=f'{parameter}_pred', color='region',
                     hover_name='id', hover_data=['context_pair', 'probe'])
    return fig

raw_type = 'psth'

@app.callback(
    Output(component_id='sample_details', component_property='figure'),
    [Input(component_id='dur_vs_amp', component_property='clickData'),
     Input(component_id='real_vs_pred_integral', component_property='clickData'),
     Input(component_id='real_vs_pred_mass_center', component_property='clickData')]
)
def _plot_sample_details(real_pic, amp_pic, dur_pic):
    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    which_clic = {'': real_pic, 'dur_vs_amp': real_pic, 'real_vs_pred_integral': amp_pic,  'real_vs_pred_mass_center':dur_pic}
    picked_eg = which_clic[trigger_id]

    cellid = picked_eg['points'][0]['hovertext']
    contexts = [int(ss) for ss in picked_eg['points'][0]['customdata'][0].split('_')]
    probes = picked_eg['points'][0]['customdata'][1]

    fig = make_subplots(1, 2)
    psth = plot_raw_pair(cellid, contexts, probes, type=raw_type)
    quant_diff = plot_time_ser_quant(cellid, contexts, probes,
                                     multiple_comparisons_axis=[1, 2], consecutive=0, cluster_threshold=0.05,
                                     fn_name='big_shuff', meta=meta_BS)

    fig.add_traces(psth['data'], rows=[1] * len(psth['data']), cols=[1] * len(psth['data']))
    fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1, row=1, col=1)
    fig.add_traces(quant_diff['data'], rows=[1] * len(quant_diff['data']), cols=[2] * len(quant_diff['data']))

    return fig

@app.callback(
    Output(component_id='model_predictions', component_property='figure'),
    [Input(component_id='dur_vs_amp', component_property='clickData'),
     Input(component_id='real_vs_pred_integral', component_property='clickData'),
     Input(component_id='real_vs_pred_mass_center', component_property='clickData')]
)
def _plot_multiple_predictions(real_pic, amp_pic, dur_pic):
    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    which_clic = {'': real_pic, 'dur_vs_amp': real_pic, 'real_vs_pred_integral': amp_pic,  'real_vs_pred_mass_center':dur_pic}
    picked_eg = which_clic[trigger_id]

    cellid = picked_eg['points'][0]['hovertext']
    contexts = [int(ss) for ss in picked_eg['points'][0]['customdata'][0].split('_')]
    probes = picked_eg['points'][0]['customdata'][1]

    fig = make_subplots(1, len(modelnames), subplot_titles=list(modelnames.keys()))

    for midx, (name, modelname) in enumerate(modelnames.items()):
        psth = plot_raw_pair(cellid, contexts, probes, type='psth', modelname=modelname, batch=batch)
        fig.add_traces(psth['data'], rows=[1] * len(psth['data']), cols=[midx+1] * len(psth['data']))
        fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1, row=1, col=midx+1)

    fig.update_layout(showlegend=False)

    return fig

@app.callback(
    Output(component_id='model_strfs', component_property='figure'),
    [Input(component_id='dur_vs_amp', component_property='clickData'),
     Input(component_id='real_vs_pred_integral', component_property='clickData'),
     Input(component_id='real_vs_pred_mass_center', component_property='clickData')]
)
def _plot_multiple_strf(real_pic, amp_pic, dur_pic):
    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    which_clic = {'': real_pic, 'dur_vs_amp': real_pic, 'real_vs_pred_integral': amp_pic,  'real_vs_pred_mass_center':dur_pic}
    picked_eg = which_clic[trigger_id]

    cellid = picked_eg['points'][0]['hovertext']
    fig = make_subplots(1, len(modelnames))

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

    return fig

@app.callback(
    Output(component_id='model_pop_mod', component_property='figure'),
    [Input(component_id='dur_vs_amp', component_property='clickData'),
     Input(component_id='real_vs_pred_integral', component_property='clickData'),
     Input(component_id='real_vs_pred_mass_center', component_property='clickData')]
)
def _plot_pop_modulation(real_pic, amp_pic, dur_pic):
    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    which_clic = {'': real_pic, 'dur_vs_amp': real_pic, 'real_vs_pred_integral': amp_pic,  'real_vs_pred_mass_center':dur_pic}
    picked_eg = which_clic[trigger_id]

    cellid = picked_eg['points'][0]['hovertext']
    contexts = [int(ss) for ss in picked_eg['points'][0]['customdata'][0].split('_')]
    probes = picked_eg['points'][0]['customdata'][1]

    fig = make_subplots(1,2)

    weight_plot = plot_pop_stategain(cellid,pop_mod, batch)
    mod_plot = plot_pop_modulation(cellid, pop_mod, batch, contexts, probes)

    fig.add_traces(weight_plot['data'], rows=[1] * len(weight_plot['data']), cols=[1] * len(weight_plot['data']))
    fig.add_traces(mod_plot['data'], rows=[1] * len(mod_plot['data']), cols=[2] * len(mod_plot['data']))

    _ = fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1,
                      row=1, col=2)

    # make common colorbar
    fig.update_layout(coloraxis=dict(colorscale='inferno',
                                     colorbar=dict(
                                         thickness=10, len=0.6,
                                         title_text='weight',
                                         title_side='right',
                                         tickangle=-90,
                                         xanchor='left', x=0.5)
                                     )
                                     )

    return fig


app.layout = html.Div([
    dcc.Graph(
        id='dur_vs_amp',
        figure=dur_vs_amp,
        clickData={'points': [{'hovertext': start_cellid,
                               'customdata': [start_ctxp, start_prb]}]}
    ),
    html.Div([dcc.Graph(id=f'real_vs_pred_{parameter}',
                        figure=plot_real_vs_model(pop_mod, parameter),
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
    dcc.Graph(
        id='model_strfs'
    ),
    dcc.Graph(
     id='model_pop_mod'
    )
])
if __name__ == '__main__':
    print("inside __name__ == '__main__'")
    app.run_server(debug=True)
