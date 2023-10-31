import pathlib as pl
import zipfile
import joblib as jl
import re

from tqdm import tqdm
import numpy as np
import scipy.stats as sst
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nems.plots.heatmap import _get_wc_coefficients, _get_fir_coefficients

from publication.globals import config, toclust_f5
from src.utils.dataframes import kruskal_with_posthoc
from src.models.modelnames import modelnames
from src.utils.subsets import cellid_A1_fit_set, cellid_PEG_fit_set, batch_map
from src.models.param_tools import load_model_xform_faster
from src.visualization.palette import CELLTYPE_COLORMAP, Black, Green
from src.visualization.interactive import plot_PSTH


def load_model_from_external_archive(cellid: str, modelname: str,
                                     archive: zipfile.ZipFile = None):
    """
    Very custom function to load model fitting context from a zip archive in an
    external hardrive. A model fit is defined by a cellid and a modelname,
    which is a strign specifying the model achitecture, in the arcane format
    defined by NEMS.

    Finds, uncompresses and returns the model fit context, which is a
    dictionary with relevant model weights, timeseries signals, figures, and
    other metadata.

    Args:
        cellid: str
        modelname: str
        archive: ZipFile archive

    Returns: dict,

    """
    if archive is None:
        print('automatically opening archive, '
              'consider passing an already open one to speed things up')
        # hardcoded for my own external drive
        archive_file = pl.Path("/media/mateo/Ext. Hard Drive Mateo/data.zip")
        archive = zipfile.ZipFile(archive_file, mode="r")

        # get hashes to extract
    file_hash = load_model_xform_faster._get_argument_hash(
        cellid=cellid, batch=batch_map[cellid], modelname=modelname
    )

    # for this specific function, the chaces in the archive are under this path
    folder_path = pl.Path(
        "data/recordings/parameters/joblib"
        "/src/models/param_tools/load_model_xform_faster/"
    )
    file_path = folder_path / file_hash / "output.pkl"
    # explicitly ensures the file is in the achive, is this really necesary?
    archive_names = set(archive.namelist())
    assert str(file_path) in archive_names

    with archive.open(str(file_path)) as pkl:
        ctx = jl.load(pkl)

    return ctx


def get_named_weights_df(ctx):
    """
    Given a model context, returns a small datadrame with information
    of the other neurons in the site, and the weight of their connection
    to the cell of interes
    Args:
        ctx: nems.ctx object

    Returns: pandas df

    """
    ms = ctx['modelspec']
    # find the postion of the stategain module
    modules = str(ms).split('\n')
    idx = modules.index('nems.modules.state.state_dc_gain')

    _ = ms.set_cell(0)
    _ = ms.set_fit(0)

    chn, npop = ms.phi[idx]['d'].shape

    mean_pop_gain = np.empty((ms.jack_count, npop))

    for jc in range(ms.jack_count):
        _ = ms.set_jack(jc)
        mean_pop_gain[jc, :] = ms.phi[idx]['d'][0,
                               :]  # drops cell first singleton dimension

    chans = ms.meta['state_chans']
    mean_pop_gain = mean_pop_gain.mean(axis=0)

    # Organizes data in a dataframe
    fnDF = pd.DataFrame([chans, mean_pop_gain],
                        index=['connection', 'weight']).T

    fnDF['id'] = ctx['meta']['cellid']
    fnDF['modelname'] = ctx['meta']['modelname']

    return fnDF


def get_strf_time_weighs_df(ctx):
    # todo document
    # Extract the strf weights into a vector over time
    ms = ctx['modelspec']

    # find first instance of sampling frequency modelname
    fs = int(re.findall('\.fs\d*\.', ms.meta['modelname'])[0][3:-1])

    wcc = _get_wc_coefficients(ms, idx=0)
    firc = _get_fir_coefficients(ms, idx=0, fs=fs)

    strf = wcc.T @ firc

    # To get a time series takes the average over spectral channels,
    # Since weights fluctuate around zero, also performs an absolute value
    # to avoid canceling out weigths and obfuscating their amplitude

    abs_ts = np.mean(np.abs(strf), axis=0)
    ts = np.mean(strf, axis=0)
    var_ts = np.var(strf, axis=0)

    # Organizes data in a dataframe
    fnDF = pd.DataFrame(
        dict(id=ctx['meta']['cellid'],
             modelname=ctx['meta']['modelname'],
             weights=(ts.tolist(),),
             absolute_weights=(abs_ts.tolist(),),
             weights_variances=(var_ts.tolist(),),
             )
    )

    return fnDF


def get_full_dataset_weights_DF(recache=False):
    """
    For every cellid belonging to the subset of neurons used for fitting,
    loads the model fit context for the full model, extracts the model weights
    for connections between neurons in the population, and the weights of the
    STRF. Finally, organizes these two sets of weights in two dataframes
    conn_df and strf_df, and merges in celltype information before cacheing.
    If the caches exist, avoids recalculation and loads the dataframes
    instead.

    Returns: two pandas dataframes (conn_df, strf_df)

    """
    # hard coded caches, TODO rename sensibly
    connection_df_file = pl.Path(
        config['paths']['analysis_cache']) / '230922_connection_DF'

    strf_time_df_file = pl.Path(
        config['paths']['analysis_cache']) / '230927_strf_time_df'

    if (connection_df_file.exists()
            and strf_time_df_file.exists()
            and not recache):
        print("cache found, loading dataframe containing "
              "all model connection weights")
        conn_DF = jl.load(connection_df_file)

        print("cache found, loading dataframe containing time STRF weights")
        strf_DF = jl.load(strf_time_df_file)
    else:
        print("Loading modelspec and organizing connection"
              " weights in dataframe and cacheing...")

        # external hard-drive with compressed data. Pre read archive for speed
        archive = zipfile.ZipFile(
            "/media/mateo/Ext. Hard Drive Mateo/data.zip", mode="r")

        all_cellids = cellid_A1_fit_set.union(cellid_PEG_fit_set)
        conn_DF = list()
        strf_DF = list()
        for cellid in tqdm(all_cellids):
            ctx = load_model_from_external_archive(
                cellid, modelnames['matchl_full'], archive=archive
            )

            conn_DF.append(get_named_weights_df(ctx))
            strf_DF.append(get_strf_time_weighs_df(ctx))

        conn_DF = pd.concat(conn_DF)
        strf_DF = pd.concat(strf_DF)

        #  data formating for memory efficientcy
        for col in ['connection', 'id', 'modelname']:
            conn_DF[col] = conn_DF[col].astype("category")
        conn_DF['weight'] = pd.to_numeric(conn_DF['weight'], downcast='float')

        for col in ['id', 'modelname']:
            strf_DF[col] = strf_DF[col].astype("category")

        jl.dump(conn_DF, connection_df_file)
        jl.dump(strf_DF, strf_time_df_file)

        print("..done")

    ##### Adds neuron type #####

    ## Twice for the sending and receiving neurons of conn_df
    # triple makes reference to the triple classification
    # (photo-)'activated', 'narrow' and 'broad'
    cell_type_df = toclust_f5.loc[:, ['id', 'triple']]

    # celltype of id (receiver)
    conn_DF = pd.merge(cell_type_df, conn_DF, on='id')
    conn_DF.rename(columns={'triple': 'receiver_type',
                            'id': 'receiver'},
                   inplace=True)

    # celltype of connection (sender)
    conn_DF = pd.merge(cell_type_df.rename(columns={'id': 'connection'}),
                       conn_DF, on='connection')
    conn_DF.rename(columns={'triple': 'sender_type',
                            'connection': 'sender'}, inplace=True)

    ## Once for the single neuron in strf_df
    strf_DF = pd.merge(
        left=toclust_f5.loc[:, ['id', 'triple']],
        right=strf_DF.loc[:, ['id', 'weights',
                              'absolute_weights', 'weights_variances']],
        on='id'
    ).rename(columns={'triple': 'neuron_type'})

    return conn_DF, strf_DF


################################## Plotting ##################################

conn_DF, strf_DF = get_full_dataset_weights_DF(recache=False)


def plot_connection_weigths_by_cell_type():
    # ToDo edit docs
    """
    Supplemental Figure M. Shows the mean and SEM for the model weight
    corresponding to putative neuronal connection (do not confuse with
    synapses).
    The weigths are classified by the neuron type of the sending and
    receiving neurons.

    Returns: Plotly Figure.

    """

    def _plot_metric_quant_bars(func_df, regressor):
        """
        Subplot function. returns a list of traces,
        each one a tick and whiskers.

        Args:
            func_df: Pandas dataframe
            regressor: str. 'sender_type', 'receiver_type'

        Returns: list of plotly traces

        """
        print(f'\n######### {regressor} #########\n')
        # Remove weights between a neuron and itself.
        # Also removes unclassified neurons, useless for this ananlsyis
        func_df = func_df.query("sender != receiver "
                                "and sender_type != 'unclass' "
                                "and receiver_type != 'unclass' ")

        _ = kruskal_with_posthoc(func_df, group_col=regressor,
                                 val_col='weight')
        func_df = func_df.groupby(regressor).agg(count=('weight', 'count'),
                                                 stat=('weight', np.mean),
                                                 err=('weight', sst.sem))
        print("summary statistics\n", func_df)

        # if you want different color error bars, have to do it one at a time
        reg_quant = list()  # list of plotly traces.
        for neu_type in ('activated', 'narrow', 'broad'):
            row = func_df.loc[neu_type]
            reg_quant.append(
                go.Scatter(x=(neu_type,), y=(row.stat,), mode='markers',
                           marker=dict(color=CELLTYPE_COLORMAP[neu_type],
                                       size=4),
                           error_y=dict(array=(row.err,),
                                        color=CELLTYPE_COLORMAP[neu_type],
                                        thickness=1, width=5),
                           showlegend=False), )

        return reg_quant

    fig = make_subplots(rows=1, cols=2,
                        shared_xaxes='columns', shared_yaxes='rows')

    for rr, reg in enumerate(['sender_type', 'receiver_type']):
        pan = _plot_metric_quant_bars(conn_DF, regressor=reg)
        fig.add_traces(pan, cols=[rr + 1] * len(pan),
                       rows=[1] * len(pan))

    h, w = 1.6, 2.9
    fig.update_layout(template='simple_white', width=96 * w, height=96 * h,
                      margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      font_family='Arial')

    fig.update_xaxes(title=dict(text="sender cell type"), col=1, row=1)
    fig.update_xaxes(title=dict(text="receiver cell type"), col=2, row=1)

    fig.update_yaxes(title=dict(text="connection weigth (AU)"),
                     col=1, row=1)

    fig.update_xaxes(title_font_size=10, title_standoff=0, tickfont_size=9)
    fig.update_yaxes(title_font_size=10, title_standoff=0, tickfont_size=9)

    return fig


def plot_strf_average_weights_over_time():
    #ToDo edit docs
    """
    Supplementary figure M pannel N. Displays a time series of the STRF
    weights, achieved by taking the average across all spectral channels.
    It uses me mean of the raw values and of the absolute values, to avoid
    possitive and negative weights canceling each other. Finaly performs this
    for all neurons used for model fit, and displays the neuron average and
    Standard error of the mean. The figure shows how there are only relevant
    weights close to the neurons responses, confirming the lack of 'linear
    information' at earlier time points and the validity of the longer STRF.

    Returns: Plotly Figure

    """
    print(f"Plotting weights for {strf_DF.shape[0]} model fits")

    fig = go.Figure()

    for ts, color in zip(
            ('absolute_weights', 'weights'),
            (Black, Green),
    ):
        plotting_array = np.stack(
            strf_DF.query("neuron_type == 'unclass'")[ts].values
        )

        _ = plot_PSTH(
            plotting_array,
            np.linspace(0, -300, 30, endpoint=False),
            CI='sem', CI_opacity=0.3,
            name=ts, showlegend=True, hist=False,
            color=color, width=1,
            fig=fig
        )

    fig.add_vline(x=-150, line=dict(width=1, dash='dot', color='black'))
    fig.add_hline(y=0, line=dict(width=1, dash='dot', color='black'))

    h, w = 1.6, 2.9
    fig.update_layout(template='simple_white', width=96 * w, height=96 * h,
                      margin=dict(l=10, r=10, t=10, b=10),
                      font_family='Arial',
                      showlegend=True,
                      legend=dict(x=0, y=1, font=dict(size=8, color="black")),
                      xaxis=dict(title=dict(text="time from response (ms)",
                                            font_size=10,
                                            standoff=0),
                                 tickfont_size=9),
                      yaxis=dict(
                          title=dict(text="STRF average weight<br>(AU)",
                                     font_size=10,
                                     standoff=0),
                          tickfont_size=9)
                      )

    return fig

def print_self_weights_statistics():
    """
    Supplementary information. Having seen the differences in connection
    weights depending on neuron type, it follows to see if neuron type also
    affects a neuron connection to itself, i.e., its adaptation pattern.
    The answer is no, there is no significant difference.

    Returns: (summary metrics, kruskal wallis, post hoc dun) objects

    """

    func_df = conn_DF.query("sender == receiver "
                            "and sender_type != 'unclass' "
                            "and receiver_type != 'unclass' ")

    kruskal, dunn = kruskal_with_posthoc(func_df, group_col='receiver_type',
                             val_col='weight')
    func_df = func_df.groupby('receiver_type').agg(count=('weight', 'count'),
                                             stat=('weight', np.mean),
                                             err=('weight', sst.sem))

    print("\nSummary statistics\n", func_df)

    return func_df, kruskal, dunn
