import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import itertools as itt

from progressbar import ProgressBar

from cpn_load import load
from cpn_reliability import signal_reliability
import cpn_dprime as cpd
import cpn_dPCA as cdPCA
import cpp_plots as cplt
import cpn_LDA as cLDA




"""
I have found a ceiling for the d' by using an optimal LDA decoder that adjusts over time (calculated independently for 
each time bin). 

the d' calculated with a dPCA projection is somehowe lower but has a similar shape

Now I need to define a floor value of d' values coming from random chance to properly asses the reduction over time of 
real d' and a lowe asimptote once contextual effects are extinguished. 

It is worth seen how correlations between neurons are changing over time and howe this affects d'
This is particularly important given that it seems that correlations themselves are the main source of discriminability
and not independent gain change, which have been acounded for by virtue of the dPCA and LDA (??? not sure about this)

for this purpose it is worth to make "blob plots" of the trial distributionse between pairs of contexts 

"""



# loads some test data

meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 0, # ms
        'raster_fs': 10,
        'transitions' : ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot' : [2,3,5,6],
        'significance': False}



code_to_name = {'t': 'Probe', 'ct': 'Context'}

site = 'AMT029a'
probe = 5
# for site, probe in zip(['AMT029a', 'ley070a'],[5,2]):

recs = load(site)
rec = recs['trip0']
sig = rec['resp']

# calculates response realiability and select only good cells to improve analysis
r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
goodcells = goodcells.tolist()

if len(goodcells) < 10:
    n_components = len(goodcells)
elif len(goodcells) == 0:
    pass  # continue
else:
    n_components = 10


####### OLD dPCA ct marginalization projection as a refference point.
Z, trialZ, significance_masks, dpca = cdPCA.tran_dpca(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                             smooth_window=meta['smoothing_window'], significance=meta['significance'],
                                             raster_fs=meta['raster_fs'])

# iterates over pairs of context transistions to calculate discriminability:
# for c0, c1 in itt.combinations(range(4),2):
c0 = 1
c1 = 3


########
# gets the real data raster (no dim reduction) to calculate single cell, population independent d'
raster = cdPCA.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                               smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'])

# trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
trialR, _, _ = cdPCA.format_raster(raster)
trialR = trialR.squeeze() # squeezes out probe
R, C, S, T = trialR.shape


# normal simulation of the data
nsim = 1000
trialN = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0), size=[nsim, C, S ,T])
trialN = trialN.squeeze()


# calculates the collection of LDA projectios
real = final_array['real']
pois = final_array['poisson']

c0 = 1
c1 = 3

x = real[:, :, c0, :]
y = real[:, :, c1, :]

lda_axes = cLDA.get_LDA_ax_over_time(x, y)
X_LDA_proj = cLDA.transform_over_time(x, lda_axes)
Y_LDA_proj = cLDA.transform_over_time(y, lda_axes)

# calcualtes d prime over the LDA projection
LDA_dprime = cpd.dprime(X_LDA_proj, Y_LDA_proj)


