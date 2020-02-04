import collections as col
import itertools as itt
import pathlib as pl
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat

import nems.epoch as nep
import nems.recording as recording
import nems_lbhb.baphy as nb
from nems import db as nd

'''
tying to import data from Sam Norman-Heineger. This data comes from experiments with the format NTI, which are a kilt of 
short varied sounds used to calculate integratione by his method (code in Matlab, not worth port in into python).

in short, this I should be alble of import the data with the appropiate tags, so I can organized and splice context-probe
pairs to use my integration analysis approach
'''

batch = 319  # NTI batch, Sam paradigm

# check sites in batch
batch_cells = nd.get_batch_cells(batch)
cell_ids = batch_cells.cellid.unique().tolist()
site_ids = set([cellid.split('-')[0] for cellid in cell_ids])

# for site in site_ids:
site = 'AMT028b'
options = {'batch': batch,
           'siteid': site,
           'stimfmt': 'envelope',
           'rasterfs': 100,
           'recache': False,
           'runclass': 'NTI',
           'stim': False}
load_URI = nb.baphy_load_recording_uri(**options)
loaded_rec = recording.load_recording(load_URI)
epochs = loaded_rec.epochs


def define_quilt_orders(folder='/auto/users/mateo/baphy/Config/lbhb/SoundObjects/@NatTempIntegrate/sounds'):
    # Get quilt orders from matfiles in folder
    order_folder = pl.Path(folder)
    orders = dict()
    for file in order_folder.iterdir():
        if re.match(r'.*order[1-2]\.mat', file.name):
            orders[file.name] = loadmat(file)
        else:
            continue

    # hardcoded values from the matlab quilt bulding software
    scram_stim_duration = 10000  # in ms
    source_stim_duration = 500  # in ms
    n_source_stim = int(round(scram_stim_duration / source_stim_duration))

    # iterates over the 12 different scrambled stims and infers the source of the segments
    formated_orders = col.defaultdict(dict)
    for scram_name, quilt_order in orders.items():

        # extract the segment duration from the scrambled stim name, formats into ms
        seg_dur = float(scram_name.split('-')[1][0:-2])

        # segment durations must be exact
        if seg_dur == 16:
            seg_dur = 15.625
        elif seg_dur == 31:
            seg_dur = 31.25
        elif seg_dur == 63:
            seg_dur = 62.50
        else:
            pass

        seg_dur = seg_dur / 1000

        # uses the order as indexes, reshapes in a 2d matrix with shape Source_stim x Start_point

        n_segs_per_sourcestim = int(quilt_order['n_segs_per_stim'].squeeze())
        order = quilt_order['order'].squeeze()
        seg_order_indices = np.sort(order).reshape([n_source_stim, n_segs_per_sourcestim])

        source_index = np.empty(order.shape)
        source_index[:] = np.nan
        source_segment = np.empty(order.shape)
        source_segment[:] = np.nan

        for ii, idx in enumerate(order):
            source_index[ii], source_segment[ii] = np.where(seg_order_indices == idx)

        formated_orders[scram_name]['order'] = order
        formated_orders[scram_name]['seg_dur'] = seg_dur
        formated_orders[scram_name]['source_index'] = source_index.astype(int)
        formated_orders[scram_name]['source_segment'] = source_segment.astype(int)

    return formated_orders


def set_epochs_from_quilts(epochs, folder=None):
    if folder is None:
        formated_orders = define_quilt_orders()
    else:
        formated_orders = define_quilt_orders(folder)

    new_epochs = epochs.copy()

    # infers pre and post stim silence from epochs
    pre = new_epochs.loc[new_epochs.name == 'PreStimSilence', ['start', 'end']].values
    pre_stim_silence = np.round(np.mean(pre[:, 1] - pre[:, 0]))

    # using the source sound and onset creates new epochs specifying sound source, segment number and duration
    for scram_name, quilt_order in formated_orders.items():
        # formats scram names to match the epochs df style
        scram_name = f"STIM_{scram_name.split('.')[0]}.wav"

        quilt = new_epochs.loc[new_epochs.name == scram_name, ['start', 'end']].values

        # prealocates an array with shape (quilt reps * segments per quilt) x 2 (start, end)
        segs_per_quilt = quilt_order['order'].size
        new_epoch_times = np.empty([segs_per_quilt * quilt.shape[0], 2])
        new_epoch_times[:] = np.nan
        new_epoch_names = list()

        # iterates over each quilt and over each segment, calculates and asign times and names
        for ii, (rep, seg) in enumerate(itt.product(range(quilt.shape[0]), range(segs_per_quilt))):
            new_epoch_times[ii, 0] = quilt[rep, 0] + pre_stim_silence + quilt_order['seg_dur'] * seg
            new_epoch_times[ii, 1] = quilt[rep, 0] + pre_stim_silence + quilt_order['seg_dur'] * (seg + 1)
            new_epoch_names.append(f"{scram_name.split('-')[1]}_"
                                   f"source-{quilt_order['source_index'][seg]}_"
                                   f"seg-{quilt_order['source_segment'][seg]}")

        df = pd.DataFrame(data=new_epoch_times, columns=['start', 'end'])
        df['name'] = new_epoch_names

        new_epochs = pd.concat([new_epochs, df], axis=0)

    # formats by sorting, index and column order
    new_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    new_epochs.reset_index(drop=True, inplace=True)

    return new_epochs


epochs = set_epochs_from_quilts(epochs)


def set_subsegments_from_epochs(epochs, regex):
    # now iteratively goes over quilt segments of longer durations and iteratively defines their "subsegments"
    # the iteration goes from longer segments to shorter ones
    # the approach asumes that eache segment is split in two subsegments of half of its duration

    new_epochs = epochs.copy()

    # finds epoch names that correspond to the specified segment duration
    matches = nep.epoch_names_matching(new_epochs, regex)
    all_names = new_epochs.loc[new_epochs.name.isin(matches), ['name']].values.squeeze()
    all_times = new_epochs.loc[new_epochs.name.isin(matches), ['start', 'end']].values

    # alocates a new array for the sub segments which should be twice as many as the source segments
    sub_seg_times = np.empty([all_names.size * 2, 2])
    sub_seg_times[:] = np.nan
    sub_seg_new_names = list()

    print(f'\ninfering subsegment epochs for {regex}, {len(all_names)} matches')

    for ss, (seg_name, seg_times) in enumerate(zip(all_names, all_times)):
        seg_dur = seg_times[1] - seg_times[0]
        new_seg_dur = seg_dur / 2
        mid_time = seg_times[0] + new_seg_dur

        # interprets and parses relevant parts of the segment name
        source = seg_name.split('_')[1]
        seg_n = int(seg_name.split('-')[-1])

        new_nom_dur = new_seg_dur * 1000
        new_nom_dur = round(new_nom_dur, 3)
        if new_nom_dur == 15.625:
            new_nom_dur = 16
        elif new_nom_dur == 31.25:
            new_nom_dur = 31
        elif new_nom_dur == 62.50:
            new_nom_dur = 63
        else:
            pass
        new_nom_dur = int(new_nom_dur)

        # first half sub_segment
        # determines name: half of the duration of the source segment, same source, segment number * 2
        first_name = f"{new_nom_dur}ms_" \
            f"{source}_" \
            f"seg-{seg_n * 2}"
        sub_seg_new_names.append(first_name)

        # determines times:
        sub_seg_times[ss * 2, 0] = seg_times[0]  # subseg start is seg start
        sub_seg_times[ss * 2, 1] = mid_time  # subseg start is midpoint

        # second half sub_segment
        second_name = f"{new_nom_dur}ms_" \
            f"{source}_" \
            f"seg-{seg_n * 2 + 1}"
        sub_seg_new_names.append(second_name)

        sub_seg_times[ss * 2 + 1, 0] = mid_time  # subseg start is midpoint
        sub_seg_times[ss * 2 + 1, 1] = seg_times[1]  # subseg end is seg end

    print(f'{sub_seg_times.shape[0]} new events ')

    df = pd.DataFrame(data=sub_seg_times, columns=['start', 'end'])
    df['name'] = sub_seg_new_names
    new_epochs = pd.concat([new_epochs, df], axis=0)

    # formats by sorting, index and column order
    new_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    new_epochs.reset_index(drop=True, inplace=True)

    return new_epochs


nom_seg_durs = [500, 250, 125, 63, 31, 16]  # nominal segment durations
for nom_seg_dur in nom_seg_durs[:-1]:
    regex = rf'{nom_seg_dur}ms_source-\d+_seg-\d+'
    epochs = set_subsegments_from_epochs(epochs, regex)


def set_contexts_from_epochs(epochs, regex):
    new_epochs = epochs.copy()

    # Now defines context probe pairs base on preexiting epochs FixMe this is an extreamly inefficiente approach.
    cp_regex = fr'C{regex}_P{regex}'
    cp_matches = nep.epoch_names_matching(new_epochs, cp_regex)

    if cp_matches:
        print(f"contexts for {regex} already present")
        return new_epochs
    else:
        pass
    p_regex = regex
    c_regex = fr'({regex})|(\APreStimSilence\Z)'

    p_matches = nep.epoch_names_matching(new_epochs, p_regex)
    probe_times = new_epochs.loc[new_epochs.name.isin(p_matches), ['name', 'start', 'end']]  # .values

    c_matches = nep.epoch_names_matching(new_epochs, c_regex)
    context_times = new_epochs.loc[new_epochs.name.isin(c_matches), ['name', 'start', 'end']]  # .values

    print(f"probe duration: {nom_seg_dur}, matched probes: {probe_times.shape[0]}")

    # gets the context for each selected probe
    cp_times = np.empty([probe_times.shape[0], 2])
    cp_times[:] = np.nan
    cp_names = list()
    for ii, (_, probe) in enumerate(probe_times.iterrows()):
        # finds the contexts which end conicides with the probe start
        context = context_times.iloc[np.where(context_times.end == probe.start)].squeeze()

        # sets the start and end times of the context probe pair
        if context['name'] == 'PreStimSilence':
            # PreStimSilence special case, cuts it so it has the same duration as the probe
            cp_times[ii, 0] = probe['start'] - (probe['end'] - probe['start'])
            cp_times[ii, 1] = probe['end']
        else:
            cp_times[ii, 0] = context['start']
            cp_times[ii, 1] = probe['end']

        new_cp_name = f"C{context['name']}_P{probe['name']}"

        cp_names.append(new_cp_name)

    # creates a df with the context prob pairs names and times and ads it to the originale epochs df
    cp_df = pd.DataFrame(data=cp_times, columns=['start', 'end'])
    cp_df['name'] = cp_names
    new_epochs = pd.concat([new_epochs, cp_df], axis=0)

    # formats by sorting, index and column order
    new_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    new_epochs.reset_index(drop=True, inplace=True)

    return new_epochs


neps = epochs.copy()
nom_seg_durs = [500, 250, 125, 63, 31, 16]  # nominal segment durations
for nom_seg_dur in nom_seg_durs:
    regex = fr'{nom_seg_dur}ms_source-\d+_seg-\d+'
    neps = set_contexts_from_epochs(neps, regex)
