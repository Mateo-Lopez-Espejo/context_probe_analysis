import pathlib as pl
from configparser import ConfigParser
from time import time

import joblib as jl
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback_context
from plotly.subplots import make_subplots

from src.root_path import config_path
from src.visualization.interactive import plot_raw_pair, plot_time_ser_quant, plot_tiling

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


# quick cache
dash_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220516_dim_red_dashboad'
recache_DF = False

# TNC014a best example for the model fitting subset
start_prb = 8 - 1
start_ctxp = '00_08'
start_cellid = 'TNC014a-22-2'
batch = 326


metrics = ['integral', 'mass_center', 'last_bin']

# subset of sites wit CPN0 and NTI
selected_sites = ['TNC014a','TNC024a']
selected_sites = ['TNC020a','TNC045a'] # odd sitese, bad??


def get_formated_DF():
    # load, filter and concat SC and PCA data. The analysis column indicates the source
    SC_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220310_ctx_mod_metric_DF_tstat_cluster_mass_BS'
    PCA_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220516_ctx_mod_metric_DF_tstat_cluster_mass_PCA'

    DF = list()
    for file in [SC_DF_file, PCA_DF_file]:
        filtered = jl.load(file).query(
            f"metric in {metrics} and mult_comp_corr == 'bf_cp' and source == 'real' and "
            f"cluster_threshold == 0.05 and value > 0 and "
            f"site in {selected_sites}")

        DF.append(filtered)

    DF = pd.concat(DF)


    # adds a PC number  column for easy filtering
    DF['PC'] = DF.id.str.extract(r'-PC-(\d+)\Z').astype(float)
    DF['PC'] = DF['PC'].fillna(value=0) # placese zero on PC for single cell, just so it can be pivoted


    # pivots for parameter space plots
    pivoted = DF.pivot_table(
        index=['analysis', 'region', 'stim_count', 'context_pair', 'probe', 'id', 'site', 'PC'],
        columns=['metric'], values='value', aggfunc='first', fill_value=0
    ).reset_index().query('last_bin > 0')

    # adds a small amount of jitter to the last bin value to help visualization
    binsize = 1 / meta['raster_fs']
    jitter = (np.random.random(pivoted.shape[0]) * binsize * 0.8 - (binsize * 0.8 / 2)) * 1000  # in ms
    pivoted['last_bin'] = pivoted['last_bin'] + jitter

    return DF, pivoted


if dash_DF_file.exists() and not recache_DF:
    print('found dash cache, loading ...')
    DF, pivoted = jl.load(dash_DF_file)
else:
    if not dash_DF_file.parent.exists():
        dash_DF_file.parent.mkdir(parents=True, exist_ok=True)

    print('loading and formatting summary dataframe')

    tic = time()
    DF, pivoted = get_formated_DF()

    print(f'it took {time() - tic:.3f}s to load and format. cacheing...')
    jl.dump([DF, pivoted], dash_DF_file)
print('done')



#### dashboard general layout
app = Dash(__name__)

toplot = pivoted.query("analysis == 'SC'").groupby(['site', 'region']).agg('mean').reset_index()

dur_vs_amp = px.scatter(data_frame=toplot, x="last_bin", y="integral", color='region',
                        hover_name='site')


def callbacks_to_input(SC_pic, PCA_pic, only_update_cp=False):
    """
    simple function to determine the last on click call back, and extract from it the necesary parameters for all
    the plotting functions
    """

    trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    # note the leading '' to catch when there has been no click at dashboard lauch
    which_click = {'': SC_pic, 'cell_scat': SC_pic, 'PC_scat': PCA_pic}
    if trigger_id not in which_click.keys():
        print('default trigger, what was it?')
        trigger_id = 'cell_scat'  # default for other callbacks not coming from the pi. this is kinda broken
    picked_eg = which_click[trigger_id]

    # special case when picking a PC. Updates the currently selected single cell, with the instance from
    # the PCA pick
    if only_update_cp:
        if trigger_id == 'PC_scat':
            cellid = SC_pic['points'][0]['hovertext']
            contexts = [int(ss) for ss in PCA_pic['points'][0]['customdata'][0].split('_')]
            probes = PCA_pic['points'][0]['customdata'][1]
            return cellid, contexts, probes

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
    siteid = real_pic['points'][0]['hovertext']

    toplot = pivoted.query(f"site == '{siteid}' and analysis == 'SC'")

    fig = px.scatter(data_frame=toplot, x="last_bin", y="integral", color='id',
                     hover_name='id', hover_data=['context_pair', 'probe'])

    print(f'\ncell scatter done, took:{time() - tic:.2f}s\n############################')
    return fig

@app.callback(
    Output(component_id='PC_scat', component_property='figure'),
    Input(component_id='dur_vs_amp', component_property='clickData'),
    Input(component_id='sel_PC', component_property='value'),
    prevent_initial_call=True
)
def plot_PC_scatter(real_pic, sel_PC):
    tic = time()
    print('############################\nplotting PCA scatter\n')
    siteid = real_pic['points'][0]['hovertext']

    toplot = pivoted.query(f"site == '{siteid}' and analysis == 'PCA' and "
                           f"PC == {sel_PC}")

    fig = px.scatter(data_frame=toplot, x="last_bin", y="integral", color='id',
                     hover_name='id', hover_data=['context_pair', 'probe'])

    print(f'\nPCA scatter done, took:{time() - tic:.2f}s\n############################')
    return fig


@app.callback(
    Output(component_id='cell_details', component_property='figure'),
    Input(component_id='cell_scat', component_property='clickData'),
    Input(component_id='PC_scat', component_property='clickData'),
    Input(component_id='raw_type', component_property='value'),
    prevent_initial_call=True
)
def _plot_sample_details(SC_pic, PC_pic, raw_type):
    tic = time()
    print('############################\nplotting cell sample details\n')
    cellid, contexts, probe = callbacks_to_input(SC_pic, PC_pic, only_update_cp=True)

    fig = make_subplots(1, 2, subplot_titles=(f'contexts: {contexts}; probe: {probe}', 'significant regions p<0.05'))
    psth = plot_raw_pair(cellid, contexts, probe, type=raw_type)
    quant_diff = plot_time_ser_quant(cellid, contexts, probe,
                                     multiple_comparisons_axis=[1, 2], consecutive=0, cluster_threshold=0.05,
                                     meta=meta)

    fig.add_traces(psth['data'], rows=[1] * len(psth['data']), cols=[1] * len(psth['data']))
    fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1, row=1, col=1)
    fig.add_traces(quant_diff['data'], rows=[1] * len(quant_diff['data']), cols=[2] * len(quant_diff['data']))

    fig.update_layout(title_text=f'{cellid}')
    print(f'\ncell sample details done, took:{time() - tic:.2f}s\n############################')
    return fig


@app.callback(
    Output(component_id='PC_details', component_property='figure'),
    Input(component_id='cell_scat', component_property='clickData'),
    Input(component_id='PC_scat', component_property='clickData'),
    Input(component_id='sel_PC', component_property='value'),
    prevent_initial_call=True
)
def _plot_PC_details(SC_pic, PCA_pic, sel_PC):
    tic = time()
    print('############################\nplotting PC sample details\n')
    cellid, contexts, probe = callbacks_to_input(SC_pic, PCA_pic)

    # transfomrs from "TNC014a-22-2" to "TNC014a-PC-1"
    cellid = f"{cellid.split('-')[0]}-PC-{sel_PC}"

    fig = make_subplots(1, 2, subplot_titles=(f'contexts: {contexts}; probe: {probe}', 'significant regions p<0.05'))
    psth = plot_raw_pair(cellid, contexts, probe, type='psth')
    quant_diff = plot_time_ser_quant(cellid, contexts, probe,
                                     multiple_comparisons_axis=[1, 2], consecutive=0, cluster_threshold=0.05,
                                     meta=meta)
    fig.add_traces(psth['data'], rows=[1] * len(psth['data']), cols=[1] * len(psth['data']))
    fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1, row=1, col=1)
    fig.add_traces(quant_diff['data'], rows=[1] * len(quant_diff['data']), cols=[2] * len(quant_diff['data']))

    fig.update_layout(title_text=f'{cellid}')
    print(f'\nPC sample details done, took:{time() - tic:.2f}s\n############################')
    return fig


# @app.callback(
#     Output(component_id='neuron_tiling', component_property='figure'),
#     Input(component_id='dur_vs_amp', component_property='clickData')
# )
# def _plot_neu_tiling(picked_eg):
#     cellid = picked_eg['points'][0]['hovertext']
#     fig = plot_tiling(cellid, filtered)
#     return fig
#
#
# @app.callback(
#     Output(component_id='site_tiling', component_property='figure'),
#     Input(component_id='dur_vs_amp', component_property='clickData')
# )
# def _plot_site_tiling(picked_eg):
#     cellid = picked_eg['points'][0]['hovertext']
#     fig = plot_tiling(cellid[:7], filtered)
#     return fig


app.layout = html.Div([
    dcc.Graph(
        id='dur_vs_amp',
        figure=dur_vs_amp,
        # clickData={'points': [{'hovertext': start_cellid,
        #                        'customdata': [start_ctxp, start_prb]}]}
    ),
    dcc.Graph(
        id='cell_scat',
    ),
    html.Div(children=[
        dcc.RadioItems(id='sel_PC', options=list(range(1,11)), value=1),
        dcc.Graph(id='PC_scat')
    ]),
    html.Div(children=[
        dcc.RadioItems(id='raw_type', options=['psth', 'raster'], value='psth'),
        dcc.Graph(id='cell_details'),
    ]),
    dcc.Graph(
        id='PC_details'
    ),
    # dcc.Graph(
    #     id='neuron_tiling'
    # ),
    # dcc.Graph(
    #     id='site_tiling'
    # )
])
if __name__ == '__main__':
    print("inside __name__ == '__main__'")
    app.run_server(debug=True)
