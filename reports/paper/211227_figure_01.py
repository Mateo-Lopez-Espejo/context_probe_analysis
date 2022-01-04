import pathlib as pl
from configparser import ConfigParser
from math import pi
import itertools as itt

import numpy as np
import scipy.stats as sst
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import  joblib as jl

from src.data.rasters import load_site_formated_raster
from src.metrics.consolidated_dprimes import _load_full_dPCA_raster, single_cell_dprimes
from src.metrics.significance import _significance

from src.visualization.fancy_plots import squarefy, quantified_dprime


# list of sequences, manually prepended zeros
sequences = np.asarray([[0,1,3,2,4,4],
                        [0,3,4,1,1,2],
                        [0,4,2,3,3,1],
                        [0,2,2,1,4,3]])

n_samps = 100
colors = ['blue', 'orange', 'green', 'purple', 'brown']
dummy_wave = np.sin(np.linspace(0,pi*4,n_samps)) # todo, pull some real example waves??
waves = [np.zeros(n_samps)] + [dummy_wave, ] * 5 + [np.zeros(n_samps)]
verrical_offset = -2
prb_idx = 3 - 1# selected probe. the -1 is to acount for 0 not being used
ctx_pair = [0,1] # pair of contexts to compare and exemplify d'
cellid = 'ARM021b-36-8'

##############################################################################
########## figure and subplots locations #####################################
##############################################################################






##############################################################################
# sound sequences plus selected examples #####################################
##############################################################################

fig, ax = plt.subplots()
for ss, seq in enumerate(sequences):
    for ww, wave_idx in enumerate(seq):
        # wave form plots
        x = np.linspace(0,1,n_samps) + ww
        y = waves[wave_idx] + ss * verrical_offset
        color = colors[wave_idx]
        ax.plot(x, y, color)

        # vertical lines for clear separation of sounds
        if ww > 0:
            ax.axvline(ww, color='black', linestyle=':')

        # add rectangle to point at exaample
        if wave_idx == prb_idx:
            rect_x = ww - 1
            rect_y = ss * verrical_offset - 1
            rect_w, rect_h = 2, 2 # 2 seconds widht, 2*norm wave
            rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                     linewidth=3, edgecolor='black', facecolor='none')
            ax.add_patch(rect)


##############################################################################
# stimulus selected examples and context type clasification ##################
##############################################################################

fig, ax = plt.subplots()
for ww, (wave, color) in enumerate(zip(waves, colors)):
    # context
    x = np.linspace(-1, 0, n_samps)
    y = wave+ ww * verrical_offset
    ax.plot(x, y, color)
    # probe
    x = np.linspace(0, 1, n_samps)
    y = waves[prb_idx] + ww * verrical_offset
    ax.plot(x, y, colors[prb_idx])

ax.axvline(0, color='black', linestyle=':')

##############################################################################
# example response ###########################################################
##############################################################################
site_raster, goodcellse = load_site_formated_raster(cellid[:7], part='all', smoothing_window=50)
eg_raster = site_raster[:, goodcellse.index(cellid),:, prb_idx, :]
fig, ax = plt.subplots()


for cxt_idx in ctx_pair:
    nsamps = eg_raster.shape[-1]
    time = np.linspace(-1, 1, nsamps)
    mean_resp = np.mean(eg_raster[:,cxt_idx, :], axis=0)
    std_resp = np.std(eg_raster[:,cxt_idx, :],axis=0)

    halfs = [np.s_[:int(nsamps/2)], np.s_[int(nsamps/2):]]
    part_color = [colors[cxt_idx], colors[prb_idx]]


    for nn, (half, color) in enumerate(zip(halfs, part_color)):

        x, y = squarefy(time[half], mean_resp[half])
        _, ystd = squarefy(time[half], std_resp[half])

        ax.plot(x, y, color=color, linewidth=3)
        if nn == 0:
            ax.fill_between(x, y-ystd, y+ystd, color=color, alpha=0.5)
        else:
            ax.fill_between(x, y-ystd, y+ystd, facecolor=color, edgecolor=part_color[0], hatch='xxx', alpha=0.5)

ax.axvline(0, color='black', linestyle=':')

##############################################################################
######## dprime and significant metrics from the example context pair ########
##############################################################################

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None,
        'stim_type': 'permutations',
        'alpha':0.05}

dprime, shuff_dprime_quantiles, goodcells, var_capt = single_cell_dprimes(cellid[:7], contexts='all', probes='all', meta=meta)
significance, confidence_interval = _significance(dprime, shuff_dprime_quantiles,
                                                  multiple_comparisons_axis=[3], consecutive=3, alpha=meta['alpha'])
cell_idx = goodcells.index(cellid) if len(cellid) > 7 else 0
pair_idx = [f'{t0}_{t1}' for t0, t1 in itt.combinations(range(dprime.shape[2]), 2)].index(f'{ctx_pair[0]}_{ctx_pair[1]}')


fig, ax = plt.subplots()
# this specific example is flipped,
fig, ax = quantified_dprime(dprime[cell_idx, pair_idx, prb_idx, :] * -1,
                            confidence_interval[:, cell_idx, pair_idx, prb_idx, :] * -1,
                            significance[cell_idx, pair_idx, prb_idx, :],
                            raster_fs=meta['raster_fs'], ax=ax)



