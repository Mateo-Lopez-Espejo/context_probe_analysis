import pathlib as pl
from configparser import ConfigParser
from time import time

import joblib as jl
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots

from src.root_path import config_path
from src.visualization.interactive import plot_raw_pair, plot_time_ser_quant, plot_tiling

#### general configuration to import the right data and caches
config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

# meta = {'reliability': 0.1,  # r value
#         'smoothing_window': 0,  # ms
#         'raster_fs': 30,
#         'montecarlo': 1000,
#         'zscore': True,
#         'stim_type': 'permutations'}
# summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220303_ctx_mod_metric_DF_tstat_cluster_mass'

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 11000,
        'zscore': True,
        'stim_type': 'permutations'}
summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220310_ctx_mod_metric_DF_tstat_cluster_mass_BS'

### same example cell as in figure 1 ###
start_prb = 3 - 1  # selected probe. the -1 is to acount for 0 not being used
start_ctxp = '00_01'  # pair of contexts to compare and exemplify d'
start_cellid = 'ARM021b-36-8'

# TNC014a best example for the model fitting subset
start_prb = 8 - 1
start_ctxp = '00_08'
start_cellid = 'TNC014a-22-2'
batch = 326

### load and preformat some of the main data
print('loading and formatting summary dataframe')
tic = time()
DF = jl.load(summary_DF_file)


# subset of sites wit CPN0 and NTI
selected_sites = ['TNC015a', 'TNC023a', 'TNC016a', 'TNC010a', 'TNC017a', 'TNC014a', 'TNC018a', 'TNC024a', 'TNC009a']

def filter_DF(DF):
    filtered = DF.query("metric in ['integral', 'last_bin'] and mult_comp_corr == 'bf_cp' and source == 'real' and "
                        "cluster_threshold == 0.05 and "
                        f"site in {selected_sites}")

    pivoted = filtered.pivot_table(index=['source', 'mult_comp_corr', 'cluster_threshold',
                                     'region', 'stim_count',
                                     'context_pair', 'probe',
                                     'id', 'site'],
                                   columns=['metric'], values='value', aggfunc='first').query('last_bin > 0').reset_index()
    # adds a small amount of jitter to the last bin value to help visualization
    binsize = 1/meta['raster_fs']
    jitter = (np.random.random(pivoted.shape[0]) * binsize * 0.8 - (binsize*0.8/2)) * 1000 # in ms
    pivoted['last_bin'] = pivoted['last_bin'] + jitter


    return filtered, pivoted

filtered, pivoted = filter_DF(DF)
print(f'it took {time() - tic:.3f}s to load')


#### dashboard general layout

app = Dash(__name__)

dur_vs_amp = px.scatter(data_frame=pivoted, x="last_bin", y="integral", color='site'
                        , hover_name='id', hover_data=['context_pair', 'probe'])

raw_type = 'psth'

@app.callback(
    Output(component_id='sample_details', component_property='figure'),
    Input(component_id='dur_vs_amp', component_property='clickData'),
    # Input(component_id='raw_type_sel', component_property=' ??? ')
)
def _plot_sample_details(picked_eg):
    print(picked_eg)

    cellid = picked_eg['points'][0]['hovertext']
    contexts = [int(ss) for ss in picked_eg['points'][0]['customdata'][0].split('_')]
    probes = picked_eg['points'][0]['customdata'][1]

    fig = make_subplots(1, 2)
    psth = plot_raw_pair(cellid, contexts, probes, type=raw_type)
    quant_diff = plot_time_ser_quant(cellid, contexts, probes,
                                     multiple_comparisons_axis=[1, 2], consecutive=0, cluster_threshold=0.05,
                                     fn_name='big_shuff', meta=meta)

    fig.add_traces(psth['data'], rows=[1] * len(psth['data']), cols=[1] * len(psth['data']))
    fig.add_vline(x=0, line_width=2, line_color='black', line_dash='dot', opacity=1, row=1, col=1)
    fig.add_traces(quant_diff['data'], rows=[1] * len(quant_diff['data']), cols=[2] * len(quant_diff['data']))

    return fig


@app.callback(
    Output(component_id='neuron_tiling', component_property='figure'),
    Input(component_id='dur_vs_amp', component_property='clickData')
)
def _plot_neu_tiling(picked_eg):
    cellid = picked_eg['points'][0]['hovertext']
    fig = plot_tiling(cellid, filtered)
    return fig


@app.callback(
    Output(component_id='site_tiling', component_property='figure'),
    Input(component_id='dur_vs_amp', component_property='clickData')
)
def _plot_site_tiling(picked_eg):
    cellid = picked_eg['points'][0]['hovertext']
    fig = plot_tiling(cellid[:7], filtered)
    return fig


app.layout = html.Div([
    dcc.Graph(
        id='dur_vs_amp',
        figure=dur_vs_amp,
        clickData={'points': [{'hovertext': start_cellid,
                               'customdata': [start_ctxp, start_prb]}]}
    ),
    dcc.Graph(
        id='sample_details'
    ),
    dcc.Graph(
        id='neuron_tiling'
    ),
    dcc.Graph(
        id='site_tiling'
    )
])
if __name__ == '__main__':
    print("inside __name__ == '__main__'")
    app.run_server(debug=True)
