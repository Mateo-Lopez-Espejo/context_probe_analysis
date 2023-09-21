import re

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.metrics.sound_spectrotemporal_analysis import calculate_sound_metrics
from src.root_path import root_path


def plot_simple_spectrogram(spectrogram):
    """
    Generates a spectrogram filling the whole figure space, with not axis ticks
    or extra information. These minimal spectrograms are used in the sound
    metrics table.
    Args:
        spectrogram: 2d numpy array

    Returns: Plotly figure

    """
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=spectrogram, coloraxis='coloraxis')
    )

    fig.update_layout(
        width=2 * 96, height=1 * 96,
        margin=dict(b=0, t=0, l=0, r=0),
        template='simple_white',
        coloraxis=dict(
            showscale=False,
            colorscale='inferno'
        ),
        xaxis=dict(
            ticks='',
            showticklabels=False,
        ),
        yaxis=dict(
            ticks='',
            showticklabels=False,
        ),
    )

    return fig


def create_sound_analysis_table():
    """
    This function generates the data to create the supplementary table 1
    describing the characteristics of the sounds used in the experiments.
    Pulls .wav files from folder, calculates spectro-temporal metrics, organizes
    in a pandas dataframe and stores in a csv file in the output file.
    Also generate figures for each sound spectrogram and store in the same
    folder. Finally, outputs a summary figure with all the spectrograms, just
    for display and reference.

    Returns: Plotly figure

    """

    in_folder = root_path / 'data' / 'sound_files'
    out_folder = root_path / 'data' / 'sound_quantifications'
    out_folder.mkdir(parents=True, exist_ok=True)

    sound_metrics_DF = list()
    panels = list()
    subtitles = list()

    for sfile in in_folder.iterdir():
        # formats the file name into a more friendly name by removing extra
        # identifiers and replacing special caracters by spaces
        sound_name = re.search(
            r'\d{2}_cat\d{2,3}_rec\d{1}_(.*)_excerpt\d{1}_\d{1}',
            str(sfile.stem)
        ).groups()[0]
        sound_name = re.sub(r'(_|-)', ' ', sound_name)
        subtitles.append(sound_name)

        (bandwidth,
         spectral_correlation,
         temporal_stationarity,
         spectrogram) = calculate_sound_metrics(sfile)

        # organizes data into a dictionary to later generate a table
        sound_metrics_DF.append({
            'name': sound_name,
            'bandwidth': round(bandwidth, 2),
            'temporal_stationarity': round(temporal_stationarity, 2),
            'spectral_correlations': round(spectral_correlation, 2),
        })

        # saves a small spectrogram that can be added to a table
        fig = plot_simple_spectrogram(spectrogram)
        panels.append(fig.data[0])
        filename = out_folder / sound_name
        fig.write_image(filename.with_suffix('.png'))

    sound_metrics_DF = pd.DataFrame(
        sound_metrics_DF
    ).sort_values(by='name').reset_index(drop=True)

    # saves the table to csv for easy edition into word file
    print(sound_metrics_DF)
    sound_metrics_DF.to_csv(out_folder / 'metric_table.csv')

    # places all spectrograms into a single multi panel figure
    # this is for reference only as the used figures are saved above.

    fig = make_subplots(rows=4, cols=4, subplot_titles=subtitles,
                        horizontal_spacing=0.01, vertical_spacing=0.05)

    for pp, pan in enumerate(panels):
        fig.add_trace(pan, row=int(np.floor(pp / 4) + 1), col=(pp % 4) + 1)

    # set color map and margins
    fig.update_layout(
        width=8 * 96, height=4 * 96,
        margin=dict(b=10, t=15, l=10, r=10),
        template='simple_white',
        coloraxis=dict(
            showscale=False,
            colorscale='inferno'
        ),
    )

    # remove ticks for cleanliness
    fig.update_xaxes(ticks='', showticklabels=False, )
    fig.update_yaxes(ticks='', showticklabels=False, )

    # reduces panel titles sizes
    fig.update_annotations(font_size=11)

    return fig
