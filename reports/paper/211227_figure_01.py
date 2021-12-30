import pathlib as pl
from configparser import ConfigParser
from math import pi

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import  joblib as jl

from src.data.rasters import load_site_formated_raster
from src.metrics.consolidated_dprimes import _load_full_dPCA_raster

# list of sequences, manually prepended and appended zeros
sequences = np.asarray([[0,1,3,2,4,4,0],
                        [0,3,4,1,1,2,0],
                        [0,4,2,3,3,1,0],
                        [0,2,2,1,4,3,0]])

n_samps = 100
colors = ['blue', 'orange', 'green', 'purple', 'brown']
dummy_wave = np.sin(np.linspace(0,pi*4,n_samps)) # todo, pull some real example waves??
waves = [np.zeros(n_samps)] + [dummy_wave, ] * 5 + [np.zeros(n_samps)]
verrical_offset = 2
rect_idx = 2 # sound around wich draw rectangles


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
        if wave_idx == rect_idx:
            rect_x = ww - 1
            rect_y = ss * verrical_offset - 1
            rect_w, rect_h = 2, 2 # 2 seconds widht, 2*norm wave
            rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                     linewidth=3, edgecolor='black', facecolor='none')
            ax.add_patch(rect)










