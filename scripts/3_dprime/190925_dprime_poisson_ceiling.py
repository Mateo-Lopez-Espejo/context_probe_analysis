import matplotlib.pyplot as plt
import numpy as np

import src.data.rasters
from src.data import LDA as cLDA, dPCA as cdPCA
from src.metrics import dprime as cpd
from src.data.load import load
from src.metrics.reliability import signal_reliability
from src.utils.tools import zscore

'''
a second attempt on the d prime approach to compare the population discriminability versus the single cell discrimination

two main points have changed since the last iteration:
1. dimention reduction defined with LDA
2. simulatede firing following a poisson distribution
3. working with z scores


End word about the script:
1. LDA indeen offers an increased d' in comparison with dPCA projections, it is an optimal decoder, and thus serves as a
cealing to the discriminability of the signal at this point

2. after thesting Normal and Poisson spike simulations, it is clear that poisson introducese certain bias. Furthermore
when binning the data with longer time windows, the spikerate tends to approach a normal distribution. I will keep using 
a normal simulation

3. Z scores:
z scores lead to a similar result as using the raw data, however the results are not as clean, and the overall d' is 
lower. this might come as a consequence of normalizing by variance which is different between cells. 
I can use this in the future but for now the raw data seems to work perfectly fine
'''

# loads some test data

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 10,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'probes_to_plot': [2, 3, 5, 6],
        'significance': False,
        'use_zscore': False}

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

# ###### OLD dPCA ct marginalization projection as a refference point.
Z, trialZ, significance_masks, dpca = cdPCA.tran_dpca(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                                      smooth_window=meta['smoothing_window'],
                                                      significance=meta['significance'],
                                                      raster_fs=meta['raster_fs'])

# iterates over pairs of context transistions to calculate discriminability:
# for c0, c1 in itt.combinations(range(4),2):
c0 = 1
c1 = 3

# gets the trial-wise projection in the top PC of the context marginalization for a pairs of contexts
# calculates population (one dimensional) d' for a pair of contexts
X_dPCA_proj = trialZ['ct'][:, 0, c0, :]
Y_dPCA_proj = trialZ['ct'][:, 0, c1, :]
dPCA_dprime = cpd.dprime(X_dPCA_proj, Y_dPCA_proj, absolute=True)

# # check that the projections and d prime make sense
# fig, ax = plt.subplots()
# ax.plot(X_dPCA_proj.T, color='green', alpha=0.3)
# ax.plot(np.mean(X_dPCA_proj, axis=0), color='green')
# ax.plot(Y_dPCA_proj.T, color='blue', alpha=0.3)
# ax.plot(np.mean(Y_dPCA_proj, axis=0), color='blue')
# ax.plot(dPCA_dprime, color='black')
# fig.suptitle(f"{site}, p{probe}, {meta['transitions'][c0]} vs {meta['transitions'][c1]}\ndPCA projection")
#

# ##### NEW LDA projectio
# real_dprime = cpd.dprime(real_proj_ctx0, real_proj_ctx1, absolute=True)

# gets the real data raster (no dim reduction) to calculate single cell, population independent d'
raster = src.data.rasters.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                          smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'])

# trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
trialR, _, _ = cdPCA.format_raster(raster)
trialR = trialR.squeeze()  # squeezes out probe

R, C, S, T = trialR.shape

nsim = 1000
# poisson simulate the data
trialP = np.random.poisson(np.mean(trialR, axis=0), size=[nsim, C, S, T])
trialP = trialP.squeeze()

# normal simulation of the data
trialN = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0), size=[nsim, C, S, T])
trialN = trialN.squeeze()

# #### check the simulated data
# cell = 17
# cont = 2
# fig, axes = plt.subplots(1,3, sharey=True, sharex=True)
# axes = np.ravel(axes)
# # real
# axes[0].plot(trialR[:,cell, cont, :].T, color='gray', alpha=0.5)
# axes[0].plot(np.mean(trialR[:,cell, cont, :], axis=0), color='black', alpha=1)
# axes[0].set_title(f'cell {cell}, context {cont}\nreal spikes')
# # normal
# axes[1].plot(trialN[:,cell, cont, :].T, color='gray', alpha=0.5)
# axes[1].plot(np.mean(trialN[:,cell, cont, :], axis=0), color='green', alpha=1)
# axes[1].set_title(f'cell {cell}, context {cont}\nnormal simulation')
# # poisson
# axes[2].plot(trialP[:,cell, cont, :].T, color='gray', alpha=0.5)
# axes[2].plot(np.mean(trialP[:,cell, cont, :], axis=0), color='blue', alpha=1)
# axes[2].set_title(f'cell {cell}, context {cont}\npoisson simulation')


all_arrays = {'real': trialR, 'normal': trialN, 'poisson': trialP}
final_array = all_arrays

# zcore the data
if meta['use_zscore']:
    # # use real mean and std (?)
    # means = np.mean(trialR, axis=(0,3))[None, :, :, None]
    # stds = np.std(trialR, axis=(0,3))[None, :, :, None]
    # zcoreR = {key: np.nan_to_num((val - means)/ stds) for key, val in all_arrays.items()}

    # use each cases mean and std
    zcoreR = {key: zscore(val, (0,2,3)) for key, val in all_arrays.items()}
    final_array = zcoreR

# checks the zscore
# cell = 17
# cont = 2
#
# fig, axes = plt.subplots(2,4, sharey=True, sharex=True)
# axes = np.ravel(axes)
#
#
# axes[0].plot(trialR[:,cell, cont, :].T, color='gray', alpha=0.5)
# axes[0].plot(np.mean(trialR[:,cell, cont, :], axis=0), color='black', alpha=1)
# axes[0].set_title(f'cell {cell}, context {cont}\nreal spikes')
#
# real = zcoreR['real']
# axes[1].plot(real[:,cell, cont, :].T, color='gray', alpha=0.5)
# axes[1].plot(np.mean(real[:,cell, cont, :], axis=0), color='black', alpha=1)
# axes[1].set_title(f'cell {cell}, context {cont}\nreal spikes')
#
# norm = zcoreR['normal']
# axes[2].plot(norm[:,cell, cont, :].T, color='gray', alpha=0.5)
# axes[2].plot(np.mean(norm[:,cell, cont, :], axis=0), color='green', alpha=1)
# axes[2].set_title(f'cell {cell}, context {cont}\nreal spikes')
#
# pois = zcoreR['poisson']
# axes[3].plot(pois[:,cell, cont, :].T, color='gray', alpha=0.5)
# axes[3].plot(np.mean(pois[:,cell, cont, :], axis=0), color='blue', alpha=1)
# axes[3].set_title(f'cell {cell}, context {cont}\nreal spikes')
#
# real = zcoreS['real']
# axes[5].plot(real[:,cell, cont, :].T, color='gray', alpha=0.5)
# axes[5].plot(np.mean(trialR[:,cell, cont, :], axis=0), color='black', alpha=1)
# axes[5].set_title(f'cell {cell}, context {cont}\nreal spikes')
#
# norm = zcoreS['normal']
# axes[6].plot(norm[:,cell, cont, :].T, color='gray', alpha=0.5)
# axes[6].plot(np.mean(norm[:,cell, cont, :], axis=0), color='green', alpha=1)
# axes[6].set_title(f'cell {cell}, context {cont}\nreal spikes')
#
# pois = zcoreS['poisson']
# axes[7].plot(pois[:,cell, cont, :].T, color='gray', alpha=0.5)
# axes[7].plot(np.mean(pois[:,cell, cont, :], axis=0), color='blue', alpha=1)
# axes[7].set_title(f'cell {cell}, context {cont}\nreal spikes')


# calculates LDA independently for each time point using charlies LDA implementation
real = final_array['real']
pois = final_array['poisson']

c0 = 1
c1 = 3

x = real[:, :, c0, :]
y = real[:, :, c1, :]

lda_axes = cLDA.fit_over_time(x, y)
X_LDA_proj = cLDA.transform_over_time(x, lda_axes)
Y_LDA_proj = cLDA.transform_over_time(y, lda_axes)

# calcualtes d prime over the LDA projection
LDA_dprime = cpd.dprime(X_LDA_proj, Y_LDA_proj)

# # check that the projections and d prime make sense
# fig, ax = plt.subplots()
# ax.plot(X_LDA_proj.T, color='green', alpha=0.3)
# ax.plot(np.mean(X_LDA_proj, axis=0), color='green', label=f'context {c0}')
# ax.plot(Y_LDA_proj.T, color='blue', alpha=0.3)
# ax.plot(np.mean(Y_LDA_proj, axis=0), color='blue', label=f'context {c1}')
# ax.plot(LDA_dprime, color='black', label="d'")
# fig.suptitle(f"{site}, p{probe}, {meta['transitions'][c0]} vs {meta['transitions'][c1]}\nLDA projection")
#
# plt.figure()
# plt.plot(LDA_dprime, color='black')
# plt.plot(dPCA_dprime, color='orange')


# projects the simulated data with dPCA and LDA  the calculate the dprimes
simulation = trialN

# dPCA
trialSR = cdPCA.transform_trials(dpca, simulation)['ct']

simX_dPCA_proj = trialSR[:, 0, c0, :]
simY_dPCA_proj = trialSR[:, 0, c1, :]

simX_dPCA_dprime = cpd.dprime(simX_dPCA_proj, simY_dPCA_proj)

# LDA
simX = simulation[:, :, c0, :]
simy = simulation[:, :, c1, :]

simX_LDA_proj = cLDA.transform_over_time(simX, lda_axes)
simY_LDA_proj = cLDA.transform_over_time(simy, lda_axes)

simX_LDA_dprime = cpd.dprime(simX_dPCA_proj, simY_dPCA_proj)

# plots every pair of stims for the real values and the simulated values, for both projection types

fig, axes = plt.subplots(2, 2)
axes = np.ravel(axes)

fog, ox = plt.subplots()

Xall = [X_dPCA_proj, X_LDA_proj, simX_dPCA_proj, simX_LDA_proj]
Yall = [Y_dPCA_proj, Y_LDA_proj, simY_dPCA_proj, simY_LDA_proj]
colors = ['black', 'orange', 'black', 'orange']
linestyles = ['-', '-', ':', ':']
labels = ['real dPCA', 'real LDA', 'sim dPCA', 'sim LDA']

for ax, X_proj, Y_proj, color, linestyle, label in zip(axes, Xall, Yall, colors, linestyles, labels):
    ax.plot(X_proj.T, color='green', alpha=0.3)
    ax.plot(np.mean(X_proj, axis=0), color='green', label=f'context {c0}')
    ax.plot(Y_proj.T, color='blue', alpha=0.3)
    ax.plot(np.mean(Y_proj, axis=0), color='blue', label=f'context {c1}')
    dprime = cpd.dprime(X_proj, Y_proj)
    ax.plot(dprime, color=color, linestyle=linestyle, label="d'")
    ax.legend()
    ox.plot(dprime, color=color, linestyle=linestyle, label=label)
ox.legend()
