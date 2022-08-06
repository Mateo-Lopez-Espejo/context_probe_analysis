from math import factorial
import itertools as itt

import numpy as np

from src.data.tensor_loaders import tensor_loaders


def pairwise_delta_FR(site, raster_meta, load_fn):
    """
    simply calculates the pairwise differences between contexts for a site array.
    A little bit of an overkill, but good to have in a single consistent place
    """

    trialR, goodcells = tensor_loaders[load_fn](site, recache_rec=False, **raster_meta)

    Rep, Neu, Ctx, Prb, Tme = trialR.shape
    R = trialR.mean(axis=0)

    Ctx_pr = int(factorial(Ctx) / (2 * factorial(Ctx - 2)))
    diffR = np.empty([Neu, Ctx_pr, Prb, Tme], dtype=float)

    for cp, (c0, c1) in enumerate(itt.combinations(range(Ctx), r=2)):
        diffR[:, cp, :, :] = R[:, c0, :, :] - R[:, c1, :, :]

    return diffR


if __name__ == "__main__":
    import plotly.express as px
    import plotly.graph_objects as go
    from src.visualization.fancy_plots import squarefy

    raster_meta = {'reliability': 0.1,  # r value
                   'smoothing_window': 0,  # ms
                   'raster_fs': 20,
                   'zscore': True,
                   'stim_type': 'permutations'}

    probe = 3
    ctx_pair = (0, 1)
    cellid = 'ARM021b-36-8'

    dfr = pairwise_delta_FR(cellid.split('-')[0], raster_meta, 'SC')
    trialR, goodcells = tensor_loaders['SC'](cellid.split('-')[0], recache_rec=False, **raster_meta)

    context_pairs = list(itt.combinations(range(trialR.shape[2]), r=2))

    y = dfr[goodcells.index(cellid), context_pairs.index(ctx_pair), probe - 1, :]

    y0 = trialR[:, goodcells.index(cellid), ctx_pair[0], probe - 1, :].mean(axis=0)
    y1 = trialR[:, goodcells.index(cellid), ctx_pair[1], probe - 1, :].mean(axis=0)

    t = np.arange(len(y))

    tq, yq = squarefy(t, y)
    _, y0q = squarefy(t, y0)
    _, y1q = squarefy(t, y1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tq, y=y0q, mode='lines', line_color='black', name=f'ctx {ctx_pair[0]}'))
    fig.add_trace(go.Scatter(x=tq, y=y1q, mode='lines', line_color='blue', name=f'ctx {ctx_pair[1]}'))
    fig.add_trace(go.Scatter(x=tq, y=yq, mode='lines', line_color='green', name=f'ctx {ctx_pair[0]} - {ctx_pair[1]}'))
    fig.show()
