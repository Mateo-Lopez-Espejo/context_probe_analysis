import numpy as np

from nems.epoch import epoch_names_matching

from src.utils.cpp_parameter_handlers import _channel_handler
from src.utils import tools


def raster_from_sig(signal, regex, channels, smooth_window=None, raster_fs=None, zscore=False):
    signal = signal.rasterize(fs=raster_fs)

    channels = _channel_handler(signal, channels)

    cp_names = epoch_names_matching(signal.epochs, regex)
    extracted = signal.extract_epochs(cp_names)

    C = len(cp_names)  # number of contexts
    P = 1  # number of probes
    R = np.min(
        [val.shape[0] for val in
         extracted.values()])  # number of repetitions
    U = len(channels)  # number of units
    T = np.max([val.shape[2] for val in extracted.values()])  # number of time bins

    raster_array = np.empty([C, P, R, U, T])
    raster_array[:] = np.nan

    for ee, (epoch, raster) in enumerate(extracted.items()):
        r = raster.shape[0]
        raster_array[ee, 0, :r, :, :] = raster[:R, channels, :]

    # defines continuous silence or sharp transitions

    probe_source = int(cp_names[0].split('_')[-2].split('-')[1])
    contexts = [name.split('_')[:-3] for name in cp_names]

    transitions = list()
    for ctx in contexts:
        if len(ctx) == 1:
            transitions.append('silence')
        else:
            context_source = int(ctx[1].split('-')[1])
            if context_source == probe_source:
                transitions.append('continuous')
            else:
                transitions.append('sharp')

    contexts = ['_'.join(ctx) for ctx in contexts]

    # takes only the second half of the raster, thus the probe
    half = int(np.ceil(raster_array.shape[-1] / 2))
    raster_array = raster_array[..., half:]

    # gaussian windows smooth
    if smooth_window is not None and smooth_window != 0:
        raster_array = tools.raster_smooth(raster_array, signal.fs, smooth_window, axis=4)

    # zscores
    if zscore is True:
        raster_array = tools.zscore(raster_array, axis=(0,1,2,4))


    return raster_array, transitions, contexts