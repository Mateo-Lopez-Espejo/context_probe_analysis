import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.mixture import GaussianMixture

import nems.db as nd

def cluster_by_metrics(DF):
    """
    takes dataframe of waveform metrics, and returns a copy of the DF with a cluster label column,
    :param DF:
    :return:
    """

    toclust = DF.dropna(axis=0).copy()
    csw = toclust['sw'].values  #csw -= csw.mean(); csw /= csw.std(); csw*=10
    cptr = toclust['ptr'].values  #cptr -= cptr.mean(); cptr /= cptr.std()
    ces = toclust['es'].values
    # wf = toclust['waveform_norm'].values
    # trough = toclust['trough'].values

    X = np.stack((csw, cptr, ces), axis=1)
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    gmm = GaussianMixture(n_components=2).fit(X)
    labels = gmm.predict(X)

    named_labels = np.empty(len(labels), dtype=object)
    if csw[labels == 1].mean() < csw[labels == 0].mean():
        named_labels[labels == 1] = 'narrow'
        named_labels[labels == 0] = 'broad'
    else:
        named_labels[labels == 0] = 'narrow'
        named_labels[labels == 1] = 'broad'

    toclust['spike_type'] = named_labels

    merge = pd.merge(DF, toclust.loc[:, ('cellid', 'spike_type')], how='left', on='cellid', validate='1:1')

    return merge


def get_waveform_metrics(mwf, flip_possitive=False):
    """
    powered by Charlie: Calculates spike width in ms (sw), peak-through ratio (ptr), full width half max in ms (fwhm), end slope (es)
    time to base line in ms (bs) and trough in index, all over a smoothed and normalized waveform (wf)
    :param mwf: 1d array of waveform,
    :return: sw, prt, fwhm, bs, trough, wf
    """
    fit2 = interpolate.UnivariateSpline(np.arange(len(mwf)), mwf)
    mwf = fit2(np.linspace(0, len(mwf), 10000))
    mwf /= abs(mwf.min())
    wf = mwf

    if mwf[np.argmax(np.abs(mwf))] > 0 and flip_possitive:
        print('flipping possitive waveform to be negative... experimental')
        mwf = -mwf

    if mwf[np.argmax(np.abs(mwf))] > 0:
        sw = ptr = fwhm = es = bs = trough = np.nan

    else:
        fs = 10000 / (82 / 30000)
        valley = np.argmin(mwf)
        peak = np.argmax(mwf[valley:]) + valley
        trough = valley

        # force 0 to be the mean of the positive waveform preceding the valley
        # mi = np.argmax(mwf[:valley]) # CHR

        # more robust manner:
        # finds the positive inflexion point (gradient == 0) prior and closest to the valley, and defines all prior as baseline
        grad = np.gradient(mwf[:valley])
        zero_grad_idx = np.where(np.diff(np.sign(grad)))[0]
        pos_inflex_idx = zero_grad_idx[grad[zero_grad_idx]>0]
        mi = pos_inflex_idx[-1] # MLE

        baseline = np.mean(mwf[:mi])
        mwf -= baseline

        sw = (peak - valley) / fs * 1000  # ms

        # get fwhm (of valley)
        left = np.argmin(np.abs(mwf[:valley] - (mwf[valley] / 2)))
        right = np.argmin(np.abs(mwf[valley:] - (mwf[valley] / 2))) + valley
        fwhm = (right - left) / fs * 1000

        if mwf[peak] <= 0:
            ptr = 0
        else:
            ptr = abs(mwf[peak]) / abs(mwf[valley])

        # 0.5 ms ~ 1800 bins
        # es.append((((mwf[valley + 850] - mwf[valley + 650])*fs) / (200 * fs)))
        # es.append((mwf[valley + 950] - mwf[valley + 900]) / 50)
        # es.append((mwf[valley + 1800] - mwf[valley + 1750]) / 50)
        # es.append((mwf[valley + 780] - mwf[valley + 750]))
        # es.append((mwf[valley + 2200] - mwf[valley + 900]))
        es = (mwf[valley + 550] - mwf[valley + 350])
        # save time waveform returns to baseline
        bs = np.argmin(np.abs(mwf[peak:])) / fs * 1000

    return sw, ptr, fwhm, es, bs, trough, wf


def get_phototags():
    """
    simple tool to list units and their corresponding phototagging,
    More things to be added later on like filtering by runclass
    """
    sql = "SELECT siteid, cellid, phototag FROM gSingleCell WHERE not(isnull(phototag))"

    d = nd.pd_query(sql)

    # formats to reduce memory usage.
    for col in d.columns:
        d[col] = d[col].astype('category')

    return d




if __name__ == '__main__':
    import plotly.express as px
    import nems_lbhb.baphy_io as io

    # # this neuron works with both approaches one is not used for some reasone
    # eg_waveform = io.get_mean_spike_waveform('TNC017a-09-1', usespkfile=False)
    # fig = px.line(y=eg_waveform)
    # fig.show()
    # eg_waveform = io.get_mean_spike_waveform('TNC017a-09-1', usespkfile=True)
    # fig = px.line(y=eg_waveform)
    # fig.show()
    #
    # # this neuron only with the legacy approach
    # eg_waveform = io.get_mean_spike_waveform('AMT020a-34-3', usespkfile=False)
    # fig = px.line(y=eg_waveform)
    # fig.show()


    # this neuron with neither
    eg_waveform = io.get_mean_spike_waveform('CRD004a-01-2', usespkfile=True)

    # sw, ptr, fwhm, es, bs, trough, wf = get_waveform_metrics(eg_waveform)
    # print(sw, ptr, fwhm, es, bs, trough)


    df = get_phototags()
    print(df)
