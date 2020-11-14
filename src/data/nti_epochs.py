import collections as col
import itertools as itt
import logging
import pathlib as pl
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat

from nems import epoch as nep
from nems.signal import TiledSignal

log = logging.getLogger(__name__)


def define_quilt_orders(folder='/auto/users/mateo/baphy/Config/lbhb/SoundObjects/@NatTempIntegrate/sounds'):
    """
    Raeads mat files used to generate the NTI sound kilts in baphy. uses this information to define the source sound and
    order in such source when creating the randomnly rearanged quilts
    :param folder: str. path to the folder containing the quilts .mat files
    :return: dict. outer key specifies the quilt name, inner key the duration, order, source and position of the
    composing segments
    """
    # Get quilt orders from matfiles in folder
    order_folder = pl.Path(folder)
    orders = dict()
    for file in order_folder.iterdir():
        if re.match(r'.*order[1-2]\.mat', file.name):
            orders[file.name] = loadmat(file)
        else:
            continue

    # hardcoded values from the matlab quilt building software
    scram_stim_duration = 10000  # in ms
    source_stim_duration = 500  # in ms
    n_source_stim = int(round(scram_stim_duration / source_stim_duration))

    # iterates over the 12 different scrambled stims and infers the source of the segments
    formated_orders = col.defaultdict(dict)
    for scram_name, quilt_order in orders.items():

        # extract the segment duration from the scrambled stim name, formats into ms
        nominal_duration = float(scram_name.split('-')[1][0:-2])

        # segment durations must be exact
        seg_dur = _nom2real_dur(nominal_duration) # in s
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


def _nom2real_dur(nominal_duration):
    if nominal_duration == 16:
        real_duration = 15.625
    elif nominal_duration == 31:
        real_duration = 31.25
    elif nominal_duration == 63:
        real_duration = 62.50
    else:
        real_duration = nominal_duration
    return real_duration


def set_epochs_from_quilts(epochs, folder=None):
    """
    gets the order of segments composing the quilts and uses them to set new epochs describing these segments
    :param epochs: pandas DataFrame. it must contain original NTI epochs
    :param folder: str. path to the folder containing the quilts .mat files
    :return: pandas DataFrame. Contains quilt-segment epochs
    """
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


def set_subsegments_from_epochs(epochs, regex):
    """
    sets sub segmets epochs from the segments specified by the regular expresion, keeps the original epochs.
    By design, the NTI segments durations are halves of the previous duration. This generatese new epochs following
    that principle
    :param epochs: pandas DataFrame. Contains segment epochs
    :param regex: str. regular expression. Specifies what segment epochs to subdivide
    :return: pandas Dataframe. Contains the original segments epochs plus the subsegmetn epochs
    """

    new_epochs = epochs.copy()

    # finds epoch names that correspond to the specified segment duration
    matches = nep.epoch_names_matching(new_epochs, regex)
    all_names = new_epochs.loc[new_epochs.name.isin(matches), ['name']].values.squeeze()
    all_times = new_epochs.loc[new_epochs.name.isin(matches), ['start', 'end']].values

    # alocates a new array for the sub segments which should be twice as many as the source segments
    sub_seg_times = np.empty([all_names.size * 2, 2])
    sub_seg_times[:] = np.nan
    sub_seg_new_names = list()

    log.info(f'infering subsegment epochs for {regex}, {len(all_names)} matches')

    for ss, (seg_name, seg_times) in enumerate(zip(all_names, all_times)):
        seg_dur = seg_times[1] - seg_times[0]
        new_seg_dur = seg_dur / 2
        mid_time = seg_times[0] + new_seg_dur

        # interprets and parses relevant parts of the segment name
        source = seg_name.split('_')[1]
        seg_n = int(seg_name.split('-')[-1])

        # transfomrs real to nominal durations, from s to ms
        new_nom_dur = round(new_seg_dur * 1000, 3)
        new_nom_dur = _real2nom_dur(new_nom_dur)

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

    log.info(f'{sub_seg_times.shape[0]} new events')

    df = pd.DataFrame(data=sub_seg_times, columns=['start', 'end'])
    df['name'] = sub_seg_new_names
    new_epochs = pd.concat([new_epochs, df], axis=0)

    # formats by sorting, index and column order
    new_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    new_epochs.reset_index(drop=True, inplace=True)

    return new_epochs


def _real2nom_dur(real_duration):
    if real_duration == 15.625:
        nominal_duration = 16
    elif real_duration == 31.25:
        nominal_duration = 31
    elif real_duration == 62.50:
        nominal_duration = 63
    else:
        nominal_duration = real_duration

    return int(nominal_duration)


def set_contexts_from_epochs(epochs, regex):
    """
    for a given set of epochs defined by regex, find the epochs inmediatly before which also match the regex.
    This creates context-probe-pair (CPP) epochs.
    :param epochs: pandas DataFrame. Should contain NTI-segment epochs, it might work with other formats
    :param regex: str. regular expresion matching the probe-epochs to which context are to be defined
    :return: pandas DataFrame. Contains CPP epochs.
    """

    new_epochs = epochs.copy()

    # Now defines context probe pairs base on preexiting epochs
    cp_regex = fr'C(({regex})|(PreStimSilence))_P{regex}'
    cp_matches = nep.epoch_names_matching(new_epochs, cp_regex)

    if cp_matches:
        log.info(f"contexts for {regex} already present")
        return new_epochs
    else:
        pass
    p_regex = regex
    c_regex = fr'({regex})|(\APreStimSilence\Z)'

    p_matches = nep.epoch_names_matching(new_epochs, p_regex)
    probe_times = new_epochs.loc[new_epochs.name.isin(p_matches), ['start', 'end', 'name']]

    c_matches = nep.epoch_names_matching(new_epochs, c_regex)
    # makes a copy to later modify the PreStimSilence values without changing the original data
    context_times = new_epochs.loc[new_epochs.name.isin(c_matches), ['start', 'end', 'name']].copy()

    log.info(f"finding contexts for probe: {p_regex}, {probe_times.shape[0]} matches")

    # modifies the start time of silence to be the same as other context sounds duration duration
    segments = probe_times.loc[:, ['start', 'end']].values
    segment_duration = np.mean(segments[:, 1] - segments[:, 0])

    context_times.loc[context_times.name == 'PreStimSilence', 'start'] = \
        context_times.loc[context_times.name == 'PreStimSilence', 'end'] - segment_duration

    # Prealocates an array for the times and for names
    cp_times = np.empty([probe_times.shape[0], 2])
    cp_times[:] = np.nan

    # finds contexts with end times equal to the probe start time
    ctx_indices = np.where(np.in1d(context_times['end'], probe_times['start']))[0]
    cp_times[:, 0] = context_times.iloc[ctx_indices, 0].values
    cp_times[:, 1] = probe_times['end'].values

    cp_names = [f"C{ctx_name}_P{prb_name}" for ctx_name, prb_name in
                zip(context_times.iloc[ctx_indices, 2], probe_times['name'])]

    # creates a df with the context prob pairs names and times and ads it to the originale epochs df
    cp_df = pd.DataFrame(data=cp_times, columns=['start', 'end'])
    cp_df['name'] = cp_names
    new_epochs = pd.concat([new_epochs, cp_df], axis=0)

    # formats by sorting, index and column order
    new_epochs.sort_values(by=['start', 'end'], ascending=[True, False], inplace=True)
    new_epochs.reset_index(drop=True, inplace=True)

    return new_epochs


def set_nti_subepochs(epochs):
    """
    wrapper function to call all previous functions for the appropriate segment durations
    :param epochs: pandas DataFrame. Should containt NTI quilt epochs
    :return: pandas DataFrame. contains initial quilt epochs plus all the derived segment and subsegment epochs
    """

    # set the segment epochs from the original quilt epochs
    epochs = set_epochs_from_quilts(epochs)

    nom_seg_durs = [500, 250, 125, 63, 31, 16]  # nominal segment durations

    # Goes over quilt segments of longer durations and iteratively defines their "subsegments"
    # the iteration goes from longer segments to shorter ones
    # the approach assumes that each segment is split in two subsegments of half of its duration
    for nom_seg_dur in nom_seg_durs[:-1]:
        regex = rf'{nom_seg_dur}ms_source-\d+_seg-\d+'
        epochs = set_subsegments_from_epochs(epochs, regex)

    # for each segment duration, defines new epochs specified context probe pairs
    for nom_seg_dur in nom_seg_durs:
        regex = fr'{nom_seg_dur}ms_source-\d+_seg-\d+'
        epochs = set_contexts_from_epochs(epochs, regex)

    return epochs


def set_signal_subepochs(signal):
    """
    Signal wrapper. set epoch names following context probe pairs
    for a signal
    :param signal: NEMS Signal
    :return: copy of the signal withe modified epochs
    """

    new_epochs = set_nti_subepochs(signal.epochs)

    # special case tiled signal copy...
    if isinstance(signal, TiledSignal):
        attributes = signal._get_attributes()
        attributes.update({'epochs': new_epochs})
        new_signal = TiledSignal(data=signal._data, safety_checks=False, **attributes)
    else:
        new_signal = signal._modified_copy(signal._data, epochs=new_epochs)
    return new_signal


def set_recording_subepochs(recording):
    """
    recording wrapper. set epoch names following context probe pairs e.g. C1_P2
    for all signals in recording and for the recording itself
    :param recording: NEMS Recording object
    :return: copy of recording with modified epochs
    """
    new_recording = recording.copy()
    for name, signal in recording.signals.items():
        new_signal = set_signal_subepochs(signal)
        new_recording[name] = new_signal
    return new_recording


def NTI_epoch_name(duration=None, source=None, position=None):
    """
    Creates an NTI compatible epoch name using durations (ms) source and source postion.
    if no parameters are passed, generates a regex that matches all NTI epochs
    :param duration: int. duration in ms
    :param source: int. idx for source sound
    :param position: int. idx for position in source sound
    :return: str.
    """

    if duration is None and source is None and position is None:
        # default regular expression, matches any NTI epoch or PreStimSilence
        return fr'\d+ms_source-\d+_seg-\d+'

    valid_durations = [500, 250, 125, 63, 31, 16]
    if duration not in valid_durations:
        raise ValueError(f'invalid duration, it must be one of {valid_durations.sort()}')

    n_sources = 20
    if source not in range(n_sources):
        raise ValueError(f'source number should be between 0 and {n_sources - 1}')

    source_duration = 500
    n_positions = int(source_duration / _nom2real_dur(duration))
    if position not in range(n_positions):
        raise ValueError(f'for this duration, positions should be between 0 and {n_positions - 1}')

    regex = rf"{duration}ms_source-{source}_seg-{position}"
    return regex