import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


import plotly.graph_objects as go

def unfold_rep_ctx_prb(arr):
    """
    Unfolds all trials, across all context and probe identities into a single dimension, while keeping information
    about these labels in an aditional dataframe. Returns an array with shape
    """
    # arr = arr.copy()# do I need a copy ??

    # Asumes a shape of the array!
    assert len(arr.shape) == 5
    rep, chn, ctx, prb, tme = arr.shape

    labels = np.empty([rep * ctx * prb, 3])
    unfolded = np.empty([rep * ctx * prb, chn, tme])

    for ii, (r, c, p) in enumerate(np.ndindex(rep, ctx, prb)):
        labels[ii,:] = np.asarray([r, c, p])
        unfolded[ii, :, :] = arr[r, :, c, p,:]

    labels = pd.DataFrame(data=labels, columns=['trial', 'context', 'probe'])

    return  unfolded, labels

def get_svm_accuracy(arr, label):
    """
    fits a linear support vector machine at each time point and
    returns the vector for the cross-validated prediction-accuracy over time.
    """
    rep, chn, tme = arr.shape # consider that here the rep dimension has all context and probes collapsed within

    accuracy = np.empty([tme])
    for tt in range(tme):
        X = arr[:,:,tt]  # n_samples x m_features == n_reps x m_units
        Y = label # n_labels. 0:rep, 1:ctx, 2:prb

        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=0.25)

        # clf = LinearSVC() # base approach, see below for normalization
        # set dual to false since we have many more trials (~400=ctx*prb*reps) than we have feature(~30 neurons)
        clf = make_pipeline(StandardScaler(), LinearSVC(dual=False, random_state=42))

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

        accuracy[tt] = accuracy_score(y_test, y_pred)

    chance = 1 / len(np.unique(Y))
    return accuracy, chance

def plot_accuracies_plotly(ctx_accuracy,ctx_chance, prb_accuracy, prb_chance, showlegend=True):

    fig = go.Figure()
    t = np.linspace(0,1,len(ctx_accuracy), endpoint=False)

    # ctx line and chance
    _ = fig.add_trace(go.Scatter(x=t, y=ctx_accuracy, mode='lines',
                                 line_color='green', name='context', showlegend=showlegend))
    _ = fig.add_trace(go.Scatter(x=[0,1], y=[ctx_chance]*2, mode='lines',
                                 line=dict(color='green',
                                           dash='dot',
                                           width=1), name='context_chance', showlegend=showlegend))
    # prb line and chance
    _ = fig.add_trace(go.Scatter(x=t, y=prb_accuracy, mode='lines',
                                 line_color='black', name='probe', showlegend=showlegend))
    _ = fig.add_trace(go.Scatter(x=[0,1], y=[prb_chance]*2, mode='lines',
                                 line=dict(color='black',
                                           dash='dot',
                                           width=1), name='context_chance', showlegend=showlegend))
    fig.update_layout(width=96*4, height=96*3,
                      margin={'l': 10, 'r': 10, 't': 10, 'b': 10, },
                      template='simple_white',
                      showlegend=True
                      )
    return fig
def decode_and_plot(arr, ploter='plotly', showlegend=True):
    """
    convenience wrapper
    """
    ufd, lbl = unfold_rep_ctx_prb(arr)
    if ploter == 'plotly':
        fig = plot_accuracies_plotly(*get_svm_accuracy(ufd, lbl['context']), *get_svm_accuracy(ufd, lbl['probe']), showlegend=showlegend)
    else:
        raise ValueError(f"ploter={ploter} unrecognized or not implemented")

    return fig, ufd, lbl
