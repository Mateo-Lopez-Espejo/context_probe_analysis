import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl

import cpn_triplets as tp
from cpn_load import load
from cpn_reliability import signal_reliability

import cpn_dPCA as cdPCA

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


# parses the relevant data from the signal object into an array of shape Context x Probe x Repetition x Unit x Time
full_array, invalid_cp, valid_cp, all_contexts, all_probes = \
    tp.make_full_array(sig, channels=goodcells, smooth_window=meta['smoothing_window'])



# selects the projection axis over which to compute the discriminability
# it can be either the top dPCA context marinalization or the axis from LDA.

# top dPCA context marginalization

Z, trialZ, significance_masks, dpca = cdPCA.tran_dpca(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                             smooth_window=meta['smoothing_window'], significance=meta['significance'],
                                             raster_fs=meta['raster_fs'])






# tow main options, eithe










