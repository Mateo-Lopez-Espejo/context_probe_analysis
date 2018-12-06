from sklearn.decomposition import PCA
import numpy as np
from nems.signal import SignalBase
import matplotlib.pyplot as plt


def signal_PCA(signal, center=True):
    if not isinstance(signal, SignalBase):
        raise TypeError('sig argument should be nems sig but is {}'.format(type(signal)))

    matrix = signal.rasterize().as_continuous().T

    # takes the mean of each cell and substracts from trace: mean response now is == 0
    if center is True:
        m = np.mean(matrix, axis=0)
        matrix = matrix - m;

    pca = PCA()
    pca.fit(matrix)
    principalComponents = pca.transform(matrix).T

    new_signal = signal.rasterize()._modified_copy(data=principalComponents, epochs=signal.epochs)
    new_signal.name = '{}_PCs'.format(signal.name)

    return new_signal, pca


def recording_PCA(recording, signal_names='all', inplace=False, center=True):
    # todo, what is the risk of making independent PCAs for different signals in the recordign?

    if signal_names == 'all':
        # if 'all' makes PCA for all the sig that are not a product of PCA
        signal_names = [sig_key for sig_key in recording.signals.keys() if sig_key.split('_')[-1] != 'PCs']
    elif isinstance(signal_names, str):
        signal_names = [signal_names]


    pca_stats = dict.fromkeys(signal_names)

    for signal_key in signal_names:
        signal = recording[signal_key]

        sig_PCs, pca = signal_PCA(signal, center=center)
        pca_stats[signal_key] = pca

        if inplace == True:
            recording.add_signal(sig_PCs)
        elif inplace == False:
            recording = recording.copy()
            recording.add_signal(sig_PCs)

    return recording, pca_stats