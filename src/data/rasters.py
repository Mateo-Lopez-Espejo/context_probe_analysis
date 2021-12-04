import itertools as itt

import numpy as np
from nems import epoch as nep
from joblib import Memory
from configparser import ConfigParser
import pathlib as pl

from src.data.load import load_with_parms
from src.metrics.reliability import signal_reliability
from src.utils import tools as tools
from src.utils.cpp_parameter_handlers import _channel_handler
from src.utils.tools import raster_smooth
from src.root_path import config_path


config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
memory = Memory(str(pl.Path(config['paths']['recording_cache']) / 'rasters'))


def _make_full_array(signal, channels='all', smooth_window=None, raster_fs=None, zscore=False):
    '''
    given a CPP/CPN signal, extract rasters and organizes in a 5D array with axes Repetition x Unit x Context x Probe x Time
    :param signal: nems signal. It should have the context probe epochs defined.
    :param channels: str, int, list, 'all'.  Defines what cells to consider
    :param smooth_window: float. gausian kernel size in ms
    :param raster_fs: int. sampling frequency of the output
    :return: nd array.
    '''

    channels = _channel_handler(signal, channels)

    signal = signal.rasterize(fs=raster_fs)

    # Zscores de data in a cell by cell manner
    if zscore:
        signal._data = tools.zscore(signal._data, axis=1)
    elif zscore is False:
        pass
    else:
        raise ValueError('meta zscore must be boolean')

    reg_ex = r'\AC\d{2}_P\d{2}'

    epoch_names = nep.epoch_names_matching(signal.epochs, (reg_ex))
    context_names = list(set([cp.split('_')[0] for cp in epoch_names]))
    context_names.sort()
    probe_names = list(set([cp.split('_')[1] for cp in epoch_names]))
    probe_names.sort()

    # gets the dimentions of the full array
    R, _, T = signal.extract_epoch(epoch_names[0]).shape
    U = len(channels)
    C = len(context_names)
    P = len(probe_names)

    full_array = np.empty([R, U, C, P, T])
    full_array[:] = np.nan

    invalid_cp = list()
    valid_cp = list()

    for pp, cc in itt.product(range(len(probe_names)), range(len(context_names))):
        cpp = '{}_{}'.format(context_names[cc], probe_names[pp])
        try:
            full_array[:, :, cc, pp, :] = signal.extract_epoch(cpp)[:, channels, :]
            valid_cp.append(cpp)
        except:
            invalid_cp.append(cpp)
            # print('{} does not exist, skipping'.format(cpp))

    # gaussian window smooth
    if smooth_window is not None and smooth_window != 0:
        print(
            'warning: smooting the data so early might lead to questionable results. Preferably smooth before plotting')
        full_array = raster_smooth(full_array, signal.fs, smooth_window, axis=4)

    return full_array, invalid_cp, valid_cp, context_names, probe_names


def __extract_triplets_sub_arr__(probes, context_types, full_array, context_names, probe_names, squeeze=True):
    '''
    short function to extract the adequate slices of the full array given a probe and the specified context transitions
    returns a copy not a view.
    :param probes: int. list. any of [2,3,5,6]
    :param context_types: str, [str,]. silence, continuous, similar, sharp. 'all' for default order of the four transitions
    :param full_array: nd array with dimensions Repetition x Unit x Context x Probe x Time
    :context_names: list of context names with order consistent with that of full array (output of make make_full_array)
    :probe_names: list of probe names with order consistent with that of full array (output of make make_full_array)
    :squeeze: bool, squeeze single value dimensions of the final array
    :return: nd array with dimensions Repetition x Unit x Context_type x Probe x Time
    '''
    ''' 
    Triplets sound order          
    array([[5, 6, 2, 3, 5],
           [6, 5, 3, 2, 6],
           [2, 4, 5, 4, 6],
           [3, 1, 2, 1, 3]])
           
    '''

    if isinstance(probes, int):
        probes = [probes]

    elif isinstance(probes, (list, tuple, set)):
        if all(isinstance(x, int) for x in probes) and all(x not in (1, 4) for x in probes):
            probes = list(probes)
        else:
            raise ValueError('all values in probe must be int and not 1 or 4')
    else:
        raise ValueError('probe must be an int or a list of ints')

    if context_types == 'all':
        context_types = ['silence', 'continuous', 'similar', 'sharp']
    elif isinstance(context_types, str):
        context_types = [context_types]
    elif isinstance(context_types, (list, tuple, set)):
        context_types = list(context_types)

    for ct in context_types:
        if ct not in ['silence', 'continuous', 'similar', 'sharp']:
            raise ValueError(f'{ct} is not a valid keyword')

    # hardcodes a map between transition names e.g. Silence, and numerated context probes e.g C0_P2
    transitions = {'P2': {'silence': 0,
                          'continuous': 1,
                          'similar': 3,
                          'sharp': 6},
                   'P3': {'silence': 0,
                          'continuous': 2,
                          'similar': 1,
                          'sharp': 5},
                   'P5': {'silence': 0,
                          'continuous': 4,
                          'similar': 6,
                          'sharp': 3},
                   'P6': {'silence': 0,
                          'continuous': 5,
                          'similar': 4,
                          'sharp': 2}}

    # create an empty array to populate with the probes slices, with contexts ordered by transition type
    # and not by contexte identity

    R, U, C, P, T = full_array.shape  # Repetition x Unit x Context x Probe x  Time

    sliced_array = np.empty([R, U, len(context_types), len(probes), T])

    for (pp, probe) in enumerate(probes):
        # find array indexes based on probe and context transition names
        p = 'P' + str(probe)
        probe_index = probe_names.index(p)

        C_names = ['C' + str(transitions[p][ct]) for ct in context_types]
        context_indices = np.asarray([context_names.index(c) for c in C_names])

        sliced_array[:, :, :, pp, :] = full_array[:, :, context_indices, probe_index, :].copy()

    if squeeze == True: sliced_array = np.squeeze(sliced_array)

    return sliced_array  # array with shape Repetitions x Unit x Context_transition x Probe x Time


def __extract_permutations_sub_arr__(probes, contexts, full_array, context_names, probe_names, squeeze=True):
    """
    Simple function to get a slice from the full array for an all_permutations experiment.
    :param probes: 'all', int or tuple of ints, probe numbers to hold
    :param contexts: 'all', int or tuple of ints, context numbers to hold
    :param full_array: nd array with dimensions Repetition x Unit x Context x Probe x Time
    :context_names: list of context names with order consistent with that of full array (output of make make_full_array)
    :probe_names: list of probe names with order consistent with that of full array (output of make make_full_array)
    :squeeze: bool, squeeze single value dimensions of the final array
    :return:  nd array with dimensions Repetition x Unit x Context x Probe x Time
    """

    # sanitize context input
    if isinstance(contexts, int):
        contexts = [contexts]
    elif contexts == 'all':
        contexts = [int(c[1:]) for c in context_names]
    elif isinstance(contexts, (list, tuple, set)):
        pass
    else:
        raise ValueError("contexts must be 'all', int or a list of ints")

    if all(isinstance(x, int) for x in contexts) and set(contexts).issubset(set(range(0, 11))):
        pass
    else:
        raise ValueError('all values in contexts must be int between 0 and 10 inclusive')


    # sanitize probe input
    if isinstance(probes, int):
        probes = [probes]
    elif probes == 'all':
        probes = [int(p[1:]) for p in probe_names if int(p[1:])!=0] # since silence does not have silence as contexts, leave it out
    elif isinstance(probes, (list, tuple, set)):
        pass
    else:
        raise ValueError("probe must be 'all' int or a list of ints")

    if all(isinstance(x, int) for x in probes) and set(probes).issubset(set(range(1, 11))):
        pass
    else:
        raise ValueError('all values in probe must be int between 1 and 10 inclusive')


    probe_indices = np.asarray([probe_names.index(f'P{p:02d}') for p in probes])
    context_indices = np.asarray([context_names.index(f'C{c:02d}') for c in contexts])

    sliced_array = np.take(full_array, context_indices, axis=2).copy()
    sliced_array = np.take(sliced_array, probe_indices, axis=3).copy()

    if squeeze: sliced_array = np.squeeze(sliced_array)

    return sliced_array


def _extract_sub_arr(probes, contexts, full_array, context_names, probe_names, stim_type, squeeze=True):
    """
    Calls the right function depending on the context_type
    """

    if stim_type == 'triplets':
        sliced_array = __extract_triplets_sub_arr__(probes, contexts, full_array, context_names, probe_names, squeeze)

    elif stim_type == 'permutations':
        sliced_array = __extract_permutations_sub_arr__(probes, contexts, full_array, context_names, probe_names,
                                                        squeeze)

    else:
        raise ValueError(f"stim_type must be 'triplets' or 'permutations' but {stim_type} was given")

    return sliced_array


def raster_from_sig(signal, probes, channels, contexts, smooth_window, raster_fs, stim_type, part='probe', zscore=False):
    full_array, invalid_cp, valid_cp, all_contexts, all_probes = \
        _make_full_array(signal, channels=channels, smooth_window=smooth_window, raster_fs=raster_fs, zscore=zscore)

    raster = _extract_sub_arr(probes=probes, contexts=contexts, full_array=full_array,
                              context_names=all_contexts, probe_names=all_probes, stim_type=stim_type, squeeze=False)

    # selects raster for context, probe or both (all)
    if part == 'probe':
        trans_idx = int(np.floor(raster.shape[-1] / 2))
        raster = raster[..., trans_idx:]
    elif part == 'context':
        trans_idx = int(np.floor(raster.shape[-1] / 2))
        raster = raster[..., :trans_idx]
    elif part == 'all':
        pass
    else:
        raise ValueError("unknonw value for 'part' parameter")

    return raster

@memory.cache
def load_site_formated_raster(site, contexts, probes, meta, part='probe', recache_rec=False):
    """
    wrapper of wrappers. Load a recording, selects the subset of data (triplets, or permutations), generates raster using
    selected  probes and transitions
    :param site: str e.g. "TNC050"
    :param contexts: list[int] or "all",
    :param probes: list[int] or "all",
    :param meta: dict with fields: "raster_fs": samps/s, "reliability" r^2, "smooth_window":ms ,
    "stim_Type": 'triplets' or 'permutations', "zscore": boolean.
    :param part: "context", "probe", "all" default "probe"
    :param recache_rec: boolean, Default False
    :return: raster with shape Repetitions x Cells x Contexts x Probes x Time_bins
    """

    recs, _ = load_with_parms(site, rasterfs=meta['raster_fs'], recache=recache_rec)
    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    # pulls the right recording depending on stimulus type and pulls the signal from it.
    if meta['stim_type'] == 'triplets':
        type_key = 'trip0'
    elif meta['stim_type'] == 'permutations':
        type_key = 'perm0'
    else:
        raise ValueError(f"unknown stim type, use 'triplets' or 'permutations'")

    sig = recs[type_key]['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_sequence*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = raster_from_sig(sig, probes=probes, channels=goodcells, contexts=contexts,
                             smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                             stim_type=meta['stim_type'],
                             zscore=meta['zscore'], part=part)
    return raster, goodcells



