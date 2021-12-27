import pathlib as pl
from configparser import ConfigParser
from math import pi

import numpy as np
import matplotlib.pyplot as plt
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


fig, ax = plt.subplots()

for ss, seq in enumerate(sequences):

    for ww, wave_idx in enumerate(seq):
        x = np.linspace(0,1,n_samps) + ww
        y = waves[wave_idx] + ss * verrical_offset
        color = colors[wave_idx]
        ax.plot(x, y, color)









