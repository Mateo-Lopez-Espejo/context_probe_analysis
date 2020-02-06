import numpy as np
import itertools as itt

import cpn_triplets as tp
from nems import epoch as nep
from nems.recording import Recording
from nems.signal import PointProcess, RasterizedSignal, TiledSignal

from cpp_parameter_handlers import _channel_handler


def raster_from_sig(signal, probe, channels, transitions, smooth_window, raster_fs=None, part='probe', zscore=False):


    full_array, invalid_cp, valid_cp, all_contexts, all_probes = \
        tp.make_full_array(signal, channels=channels, smooth_window=smooth_window, raster_fs=raster_fs)

    raster = tp.extract_sub_arr(probes=probe, context_types=transitions, full_array=full_array,
                                context_names=all_contexts, probe_names=all_probes, squeeze=False)

    # selects raster for context, probe or both (all)
    if part == 'probe':
        trans_idx = int(np.floor(raster.shape[-1]/2))
        raster = raster[..., trans_idx:]
    elif part == 'context':
        trans_idx = int(np.floor(raster.shape[-1]/2))
        raster = raster[..., :trans_idx]
    elif part == 'all':
        pass
    else:
        raise ValueError("unknonw value for 'part' parameter")

    # Zscores de data in a cell by cell manner
    if zscore is True:
        mean = np.mean(raster, axis=(0,1,2,4))[None, None, None, :, None]
        std = np.std(raster, axis=(0,1,2,4))[None, None, None, :, None]
        raster = np.nan_to_num((raster - mean) / std)
    elif zscore is False:
        pass
    else:
        raise ValueError('meta zscore must be boolean')

    return raster