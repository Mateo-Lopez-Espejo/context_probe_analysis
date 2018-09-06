from sklearn.decomposition import PCA
import numpy as np
import nems.signal
import pandas as pd
from nems.signal import SignalBase


def signal_PCA(signal, center=True):
    if not isinstance(signal, SignalBase):
        raise TypeError('sig argument should be nems sig but is {}'.format(type(signal)))

    matrix = signal.rasterize().as_continuous().T

    if center is True:
        m = np.mean(matrix, axis=0)
        matrix = matrix - m;

    pca = PCA()
    pca.fit(matrix)
    principalComponents = pca.transform(matrix).T

    new_signal = signal.rasterize()._modified_copy(data=principalComponents, epochs=signal.epochs)
    new_signal.name = '{}_PCs'.format(signal.name)

    return new_signal, pca


def charlie_PCA(signal, center=True):
    """
    computes pca on the input matrix r.
    r can also be a nems sig, either will work.

    output is a dictionary containing the PCs, variance explained per net pcs, stepwise
    variance explained per individual pc, and the loading (rotation) matrix.
    """
    if not isinstance(signal, SignalBase):
        raise TypeError('sig argument should be nems sig but is {}'.format(type(signal)))

    r_pca = signal.rasterize().as_continuous()

    if center is True:
        m = np.mean(r_pca, axis=0)  # is this a bug?? the mean should be done over eacheneuron, not over each time bin
        r_pca = r_pca - m;

    r_pca = r_pca.T

    U, S, V = np.linalg.svd(r_pca, full_matrices=False)
    v = S ** 2
    step = v
    var_explained = []
    for i in range(0, r_pca.shape[1]):
        var_explained.append(100 * (sum(v[0:(i + 1)]) / sum(v)))
    loading = V
    pcs = U.T  # *S;

    out = {'pcs': pcs,
           'variance': var_explained,
           'step': step,
           'loading': loading}

    new_signal = signal.rasterize()._modified_copy(pcs, epochs=signal.epochs)
    new_signal.name = '{}_PCs'.format(new_signal.name)

    return new_signal, out


def recording_PCA(recording, signal_names='all', inplace=False, method='sklearn', center=True):
    # todo, what is the risk of making independent PCAs for different signals in the recordign?

    if signal_names == 'all':
        # if 'all' makes PCA for all the sig that are not a product of PCA
        signal_names = [sig_key for sig_key in recording.signals.keys() if sig_key.split('_')[-1] != 'PCs']
    elif isinstance(signal_names, str):
        signal_names = [signal_names]


    pca_stats = dict.fromkeys(['{}_PCs'.format(sig_key) for sig_key in signal_names])

    for signal_key in signal_names:
        signal = recording[signal_key]


        # todo clear the mess of  this two dispair methods.
        if method == 'sklearn':
            sig_PCs, pca = signal_PCA(signal, center=center)
            pca_stats['{}_PCs'.format(signal_key)] = {'step':pca.explained_variance_}
        elif method == 'charlie':
            sig_PCs, pca_stats['{}_PCs'.format(signal_key)] = charlie_PCA(signal, center=center)

        if inplace == True:
            recording.add_signal(sig_PCs)
        elif inplace == False:
            recording = recording.copy()
            recording.add_signal(sig_PCs)

    return recording, pca_stats