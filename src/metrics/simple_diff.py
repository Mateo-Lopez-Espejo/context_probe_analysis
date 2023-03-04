import itertools as itt

import numpy as np
import xarray as xr
import pandas as pd

def ctx_effects_as_DF(arr, cellids, fs):
    # replace context by context-pair derived difference
    rep, chn, ctx, prb, tme = arr.shape
    ctx_pairs = list(itt.combinations(range(ctx), 2))

    diff_arr = np.empty([rep, chn, len(ctx_pairs), prb, tme])

    for cpidx, (c0, c1) in enumerate(ctx_pairs):
        diff_arr[:, :, cpidx, :, :] = arr[:, :, c0, :, :] - arr[:, :, c1, :, :]

    #### turns into xarray for later transofmration into dataframe, consider the dimension names and coordinates
    # will be columns and col values in the dataframe
    diff_arr = xr.DataArray(diff_arr, dims=('rep', 'id', 'context_pair', 'probe', 'time'),
                coords={'id': cellids,
                        'context_pair': [f'{c1:02d}_{c2:02d}' for c1, c2 in itt.combinations(range(ctx), 2)],
                        'probe': range(1, prb + 1),
                        'time': np.linspace(0, tme/fs, tme, endpoint=False)})

    chnk = int(tme/4)
    time_chunks = {'A': np.s_[..., :chnk], 'B': np.s_[..., chnk:2*chnk],
                   'C': np.s_[..., 2*chnk:3*chnk], 'D': np.s_[..., 3*chnk:],
                   'Full': np.s_[..., :]}

    IDF = list()

    for chunk_name, slicer in time_chunks.items():
        # classic absolute integral
        # idf = np.abs(diff_arr[slicer].mean(dim='rep')).sum(dim='time') * 1/raster_meta['raster_fs']
        idf = diff_arr[slicer].mean(dim='rep').sum(dim='time') / fs # plain integral
        # idf = (diff_arr[slicer].mean(dim='rep')**2).mean(dim='time') # mean square error across time

        idf = idf.to_dataframe(name='value').reset_index() # name of the values  column in long format

        idf['stim_count'] = prb
        idf['chunk'] = chunk_name
        IDF.append(idf)

    return pd.concat(IDF, ignore_index=True, axis=0)