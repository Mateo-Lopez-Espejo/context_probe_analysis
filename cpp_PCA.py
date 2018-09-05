from sklearn.decomposition import PCA
import numpy as np
import nems.signal
import pandas as pd
from nems.signal import SignalBase


def signal_PCA(signal):
    if not isinstance(signal, SignalBase):
        raise TypeError('signal argument should be nems signal but is {}'.format(type(signal)))

    matrix = signal.rasterize().as_continuous().T
    pca = PCA()
    principalComponents = pca.fit_transform(matrix).T

    new_signal = signal.rasterize()._modified_copy(data=principalComponents, epochs=signal.epochs)
    new_signal.name = '{}_PCs'.format(signal.name)

    return new_signal


def recording_PCA(recording, signal_names='all', inplace=False):
    if signal_names == 'all':
        signal_names = list(recording.signals.keys())

    for signal_key in signal_names:
        signal = recording[signal_key]

        sig_PCs = signal_PCA(signal)

        if inplace == True:
            recording.add_signal(sig_PCs)
        elif inplace == False:
            recording = recording.copy()
            recording.add_signal(sig_PCs)

    return recording


def charlie_PCA(signal, center=True):
    """
    computes pca on the input matrix r.
    r can also be a nems signal, either will work.

    output is a dictionary containing the PCs, variance explained per net pcs, stepwise
    variance explained per individual pc, and the loading (rotation) matrix.
    """
    if not isinstance(signal, SignalBase):
        raise TypeError('signal argument should be nems signal but is {}'.format(type(signal)))

    r_pca = signal.rasterize().as_continuous()

    if center is True:
        m = np.mean(r_pca, axis=0)
        r_pca = r_pca - m;

    if r_pca.shape[0] < r_pca.shape[1]:
        r_pca = r_pca.T

    U, S, V = np.linalg.svd(r_pca, full_matrices=False)
    v = S ** 2
    step = v
    var_explained = []
    for i in range(0, r_pca.shape[1]):
        var_explained.append(100 * (sum(v[0:(i + 1)]) / sum(v)))
    loading = V
    pcs = U  # *S;

    out = {'pcs': pcs,
           'variance': var_explained,
           'step': step,
           'loading': loading}

    new_signal = signal.rasterize()._modified_copy(pcs, epochs=signal.epochs)
    new_signal.name = '{}_PCs'.format(new_signal.name)

    return new_signal, out
