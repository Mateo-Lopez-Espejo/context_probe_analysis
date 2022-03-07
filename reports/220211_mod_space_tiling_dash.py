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
from src.visualization.interactive import plot_psth_pair, plot_time_ser_quant

#### general configuration to import the right data and caches
config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'stim_type': 'permutations'}
summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220303_ctx_mod_metric_DF_tstat_cluster_mass'

### same example cell as in figure 1 ###
start_prb = 3 - 1  # selected probe. the -1 is to acount for 0 not being used
start_ctxp = '00_01'  # pair of contexts to compare and exemplify d'
start_cellid = 'ARM021b-36-8'

### load and preformat some of the main data
print('loading and formatting summary dataframe')
tic = time()
DF = jl.load(summary_DF_file)
print(f'it took {time() - tic:.3f}s to load')


def format_dataframe(DF):
    tic = time()
    ff_analylis = DF.analysis.isin(['SC'])
    ff_badsites = ~DF.siteid.isin(['TNC010a'])
    mask = ff_analylis & ff_badsites

    if 'cluster_threshold' not in DF.columns:
        DF['cluster_threshold'] = 0

    good_cols = ['source', 'mult_comp_corr', 'cluster_threshold', 'region', 'siteid', 'cellid', 'context_pair',
                 'probe', 'metric', 'value']
    filtered = DF.loc[mask, good_cols]
    print(f'first pass filtering {time() - tic:.3f}s ')

    filtered['probe'] = [int(p) for p in filtered['probe']]
    filtered['context_pair'] = [f"{int(cp.split('_')[0]):02d}_{int(cp.split('_')[1]):02d}"
                                for cp in filtered['context_pair']]
    print(f'changing ctx prb name {time() - tic:.3f}s ')

    # rename metrics and analysis for ease of ploting
    filtered['metric'] = filtered['metric'].replace({'significant_abs_mass_center': 'duration',
                                                     'significant_abs_sum': "amplitude"})
    print(f'changing metric names {time() - tic:.3f}s ')

    filtered['id'] = filtered['cellid'].fillna(value=filtered['siteid'])
    filtered = filtered.drop(columns=['cellid'])
    filtered.rename(columns={'siteid': 'site'}, inplace=True)
    print(f'consolidating ids {time() - tic:.3f}s ')

    filtered['value'] = filtered['value'].fillna(value=0)
    print(f'dealing with non significant{time() - tic:.3f}s ')

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
    print(f'defining transition type {time() - tic:.3f}s ')

    # column specifying number of different sounds used
    nstim = filtered.groupby(['id']).agg(stim_count=('probe', lambda x: x.nunique()))
    filtered = pd.merge(filtered, nstim, on='id')
    print(f'defining n stimuli {time() - tic:.3f}s ')

    return filtered


DF_long = format_dataframe(DF)

pivoted = DF_long.pivot_table(index=['source', 'mult_comp_corr', 'cluster_threshold',
                                     'region', 'stim_count',
                                     'context_pair', 'probe',
                                     'id', 'site'],
                              columns=['metric'], values='value', aggfunc='first')

# frankesntein indexing. first get only the real data, then dismiss non significant point
dur_vs_amp_df = pivoted.loc[('real', 'none', 0.01), :].query('duration >  0').reset_index()

### simplified version of the tile figure ###
# prefilters all the data to be ploted (multiple neurons) to ensure shared colormaps
# for the fill values

#### dashboard general layout

app = Dash(__name__)

dur_vs_amp = px.scatter(data_frame=dur_vs_amp_df, x="duration", y="amplitude", color='region'
                        , hover_name='id', hover_data=['context_pair', 'probe'])

app.layout = html.Div([
    dcc.Graph(
        id='dur_vs_amp',
        figure=dur_vs_amp,
        clickData={'points': [{'hovertext': start_cellid,
                               'customdata': [start_ctxp, start_prb]}]}
    ),
    dcc.Graph(
        id='sample_details'
    )
])


@app.callback(
    Output(component_id='sample_details', component_property='figure'),
    Input(component_id='dur_vs_amp', component_property='clickData')
)
def _plot_sample_details(picked_eg):
    print(picked_eg)

    cellid = picked_eg['points'][0]['hovertext']
    contexts = [int(ss) for ss in picked_eg['points'][0]['customdata'][0].split('_')]
    probes = picked_eg['points'][0]['customdata'][1]

    fig = make_subplots(1, 2)

    psth = plot_psth_pair(cellid, contexts, probes)
    quant_diff = plot_time_ser_quant(cellid, contexts, probes,
                                     multiple_comparisons_axis=[1, 2], consecutive=0, cluster_threshold=0.05,
                                     fn_name='t_statistic', meta=meta)

    fig.add_traces(psth['data'], rows=[1] * len(psth['data']), cols=[1] * len(psth['data']))
    fig.add_traces(quant_diff['data'], rows=[1] * len(quant_diff['data']), cols=[2] * len(quant_diff['data']))

    return fig


if __name__ == '__main__':
    print('inside nane == main')
    app.run_server(debug=True)
