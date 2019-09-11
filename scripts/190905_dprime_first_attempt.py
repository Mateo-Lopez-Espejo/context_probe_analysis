import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl

import cpn_triplets as tp
from cpn_load import load
from cpn_reliability import signal_reliability
import cpn_dprime as cpd

import cpn_dPCA as cdPCA

import itertools as itt



from scipy.stats import gaussian_kde as gkde
import seaborn as sn
import cpp_plots as plots


"""
Previous iterations on the euclidean distance analysis were unadecuate to determine whether population codes helped 
discrimination of contexts when compared with single cell representations.

Here I am using a Discrimination index (d prime, d') sugested by charlie to compare single cell vs population discrimination

The population discriminations is defined over the projection to a one dimentional axis (selected through LDA or dPCA)

The single cell discriminations is defined as the ndminetional pitagorean hypothenuse of discrimination across every 
single cell dimension.

"""


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',
                  '#984ea3', '#999999', '#e41a1c', '#dede00']

# meta parameter
meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 20, # ms
        'raster_fs': 100,
        'transitions' : ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot' : [2,3,5,6],
        'significance': False}



code_to_name = {'t': 'Probe', 'ct': 'Context'}

save_img = False


site = 'AMT029a'
probe = 5


for site, probe in zip(['AMT029a', 'ley070a'],[5,2]):

    recs = load(site)
    rec = recs['trip0']
    sig = rec['resp'].rasterize()

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])

    goodcells = goodcells.tolist()

    if len(goodcells) < 10:
        n_components = len(goodcells)
    elif len(goodcells) == 0:
        pass  # continue
    else:
        n_components = 10

    # selects the projection axis over which to compute the discriminability
    # it can be either the top dPCA context marginalization, which is an axis stable over time,
    # or the axis from LDA, which is defined for each time bin

    # top dPCA context marginalization.
    Z, trialZ, significance_masks, dpca = cdPCA.tran_dpca(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                                 smooth_window=meta['smoothing_window'], significance=meta['significance'],
                                                 raster_fs=meta['raster_fs'])


    # iterates over pairs of context transistions to calculate discriminability:

    # for c0, c1 in itt.combinations(range(4),2):

    c0 = 1
    c1 = 3

    # gets the trial-wise projection in the top PC of the context marginalization
    ctx_proj = trialZ['ct'][:, 0, :, :]

    # calculates population (one dimensional) d' for a pair of contexts
    ctx0 = ctx_proj[:, c0, :]
    ctx1 = ctx_proj[:, c1, :]

    dprime = cpd.dprime(ctx0, ctx1,absolute=True)

    # calculates single cell (n dimensional) d'


    raster = cdPCA.raster_from_sig(sig,probe,channels=goodcells, transitions=meta['transitions'],
                                       smooth_window=meta['smoothing_window'],raster_fs=meta['raster_fs'])


    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, centers = cdPCA.format_raster(raster)

    ctx0 = trialR[:, :, 1, 0, :]
    ctx1 = trialR[:, :, 3, 0, :]

    ndim_dprime = cpd.ndim_dprime(ctx0, ctx1, absolute=True)

    fig, ax = plt.subplots()

    ax.plot(dprime, color='green', label='population dprime')
    ax.plot(ndim_dprime, color='black', label='single units dprime')
    fig.suptitle(f"{site}, probe {probe}; {meta['transitions'][c0]} vs {meta['transitions'][c1]}")

