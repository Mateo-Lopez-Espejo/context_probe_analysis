import nems.recording as recording
import nems_lbhb.baphy as nb
import nems.preprocessing as preproc
import nems.epoch as nep
import src.metrics.distance

from src.data import epochs as cpe, cache as cch, PCA as cpca, rasters as tp
from src.metrics import prm_dispersion as cdisp
from src.visualization import fancy_plots as cplot

import  numpy as np
import scipy.stats as sst
import itertools as itt
import matplotlib.pyplot as plt

'''
first attempt of addapting the old CPP analisie to the new datasets using runclass CPN (context probe natural sound)
CPN contains two main variationse "triplets" and "all permutations".

1. Triplets: the second iteration on CPP, uses adyacent snippets of sounds and shuffles theirs order. the purpose o to
study how different or similar statistics from the context in relation to the probe can influence more or less the probe 
response

2. All permutations: variation on the first iteration of CPP, instead of using artificial vocalization, it uses 4 
different natural sounds

this was a first pass with no discrimination of context transitions. a good startign point to see the raw data.
'''

# find cells/site
# CPN

site = 'ley070a' # good site. A1
site = 'ley072b' # Primary looking responses with strong contextual effects
site = 'AMT028b' # good site
site = 'AMT029a' # Strong response, somehow visible contextual effects
site = 'AMT030a' # low responses, Ok but not as good
site = 'AMT031a' # low response, bad
site = 'AMT032a' # great site. PEG


# ToDo check visual inspection
# ley070a
goodcells = ['ley070a-01-1', 'ley070a-01-2', 'ley070a-07-1', 'ley070a-12-3', 'ley070a-18-2', 'ley070a-19-1',
             'ley070a-19-2', 'ley070a-23-1', 'ley070a-27-1', 'ley070a-37-1', 'ley070a-42-1']
best_cell = 'ley070a-37-1'

# ley072b
goodcells = ['ley072b-11-1', 'ley072b-18-1', 'ley072b-25-1', 'ley072b-27-1', 'ley072b-32-1', 'ley072b-34-1']
best_cell = 'ley072b-25-1'

# AMT028b
goodcells = ['AMT028b-06-1', 'AMT028b-13-1', 'AMT028b-20-1', 'AMT028b-22-1', 'AMT028b-28-2', 'AMT028b-30-1',
             'AMT028b-34-1', 'AMT028b-37-2', 'AMT028b-40-1', 'AMT028b-43-1', 'AMT028b-48-1', 'AMT028b-55-1']
best_cell = 'AMT028b-40-1'

# AMT029a
goodcells = ['AMT029a-09-1', 'AMT029a-10-1', 'AMT029a-19-1', 'AMT029a-23-1', 'AMT029a-26-1', 'AMT029a-30-1',
             'AMT029a-35-1', 'AMT029a-40-1', 'AMT029a-40-3', 'AMT029a-43-1', 'AMT029a-49-1', 'AMT029a-52-1']
best_cell = 'AMT029a-49-1'

# AMT030a
goodcells = ['AMT030a-21-1', 'AMT030a-21-3', 'AMT030a-22-1', 'AMT030a-22-2', 'AMT030a-24-1', 'AMT030a-27-1',
             'AMT030a-27-2', 'AMT030a-28-1', 'AMT030a-28-3', 'AMT030a-30-2']
best_cell = 'AMT030a-21-3'

# AMT031a
goodcells = ['AMT031a-03-1', 'AMT031a-20-1', 'AMT031a-50-1', 'AMT031a-51-1', 'AMT031a-53-1', 'AMT031a-57-2']
best_cell = 'AMT031a-50-1'

# AMT032a
goodcells = ['AMT032a-23-1', 'AMT032a-26-1', 'AMT032a-28-1', 'AMT032a-32-1', 'AMT032a-39-1', 'AMT032a-45-2',
             'AMT032a-49-1', 'AMT032a-51-2', 'AMT032a-55-1']
best_cell = 'AMT032a-51-2'


modelname = 'resp'
options = {'batch': 316,
           'siteid': site,
           'stimfmt': 'envelope',
           'rasterfs': 100,
           'recache': False,
           'runclass': 'CPN',
           'stim': False}  #ToDo chace stims, spectrograms???

load_URI = nb.baphy_load_recording_uri(**options)
loaded_rec = recording.load_recording(load_URI)

# load experimetn params # ToDo, proper experimet  read
# rename epochs taking into account source: CPN and CPPv2 (ie. triplests and all permutations)
rec = cpe.set_recording_subepochs(loaded_rec)
sig = rec['resp']
eps = sig.epochs

cellorder = sig.chans
# map of transitin types, which change form probe to probe. the outer key is probe, the inner key is transition type
# the value is context identity
transitions = {6: {'silence': 0,'continuous': 5,'similar': 7,'sharp': 10},
               7: {'silence': 0,'continuous': 6,'similar': 5,'sharp': 9},
               9: {'silence': 0,'continuous': 8,'similar': 10,'sharp': 7},
               10: {'silence': 0,'continuous': 9,'similar': 8,'sharp': 6}}

########################################################################################################################

# check integrity of the data
# test plot
fig, axes = cplot.hybrid(sig, ['C0_P6', 'C0_P7', 'C0_P9', 'C0_P10'])
fig, axes = cplot.hybrid(sig, epoch_names = ['REFERENCE'])
fig, axes = cplot.hybrid(sig, epoch_names = ['REFERENCE'], channels=goodcells)

# data plots withe this approach corresponds with that plotted with baphy_remote. It seems that the epoch naming is
# properly aligned.
epoch_names = nep.epoch_names_matching(sig.epochs, r'\ASTIM_Tsequence.*')
fig, axes = cplot.hybrid(sig, epoch_names=epoch_names, channels=goodcells)
fig, axes = cplot.hybrid(sig, epoch_names='C7_P9', channels=goodcells)
fig, axes = cplot.hybrid(sig, epoch_names='C0_P9', channels=goodcells)

# for good cells, all the relevant probes after silence
fig, axes = cplot.hybrid(sig, epoch_names=r'\AC0_P([679]|10)\Z', channels=goodcells)
fig, axes = cplot.hybrid(sig, epoch_names=r'\AC0_P([679]|10)\Z', channels=best_cell)

# best cell, best probe, all the contexts
fig, axes = cplot.hybrid(sig, epoch_names=r'\AC(\d|10)_P6\Z', channels=goodcells)
fig, axes = cplot.hybrid(sig, epoch_names=r'\AC(\d|10)_P6\Z', channels=best_cell)



########################################################################################################################

# organizes relevant data in array with dimensions Context x Probe x Repetition x Unit x Time
full_array, bad_cpp, good_cpp, context_names, probe_names = tp.make_full_array(sig, 'CPN')

# now calculate pairwise difference between context types
valid_probes = [6, 7, 9, 10]
context_transitions = ['silence', 'continuous', 'similar', 'sharp']
diff_arr = src.metrics.distance.pairwise_PSHT_distance(valid_probes, context_transitions, full_array, context_names, probe_names, )

########################################################################################################################
# plot the PSTHs of a probe given two contexts transitions, compares,

p = 7
ct1 = 'continuous'
ct2 = 'sharp'
cell = cellorder.index(best_cell)

arr1 = tp._extract_triplets_sub_arr(p, ct1, full_array, context_names, probe_names) # shape Rep x Unit x Time
arr2 = tp._extract_triplets_sub_arr(p, ct2, full_array, context_names, probe_names)

psth1 = np.mean(arr1, axis=0) # shape Unit x Time
psth2 = np.mean(arr2, axis=0)
diff = np.absolute(psth1 - psth2)
SEM1 = sst.sem(arr1, axis=0)
SEM2 = sst.sem(arr2, axis=0)
significance = diff > (SEM1 + SEM2)
sig_start, sig_end = cplot._sig_bin_to_time(significance, window=1, fs=100, unit_overlaping=True)
timeax = np.linspace(0,2, psth1.shape[1])


fig, ax = plt.subplots()
ax.plot(timeax, psth1[cell, :], color='blue')
ax.plot(timeax, psth2[cell, :], color='orange')
ax.plot(timeax, diff[cell, :], color='black')

ax.fill_between(timeax, psth1[cell, :] - SEM1[cell,:], psth1[cell, :] + SEM1[cell,:],
                color='blue', alpha=0.5)
ax.fill_between(timeax, psth2[cell, :] - SEM2[cell, :], psth2[cell, :] + SEM2[cell, :],
                color='orange', alpha=0.5)

ax = cplot._significance_bars(sig_start[cell], sig_end[cell], ax=ax)


# check that the difference is correctly calculatede and is consisten with the previous block of code
probe = valid_probes.index(p)
ctrans1 = context_transitions.index(ct1)
ctrans2 = context_transitions.index(ct2)

test = diff_arr[probe, ctrans1, ctrans2, cell, :, :]
timeax = np.linspace(0, 2, test.shape[0])
fig, ax = plt.subplots()
ax.plot(timeax, test[:, 0], color='black')

sig_start, sig_end = cplot._sig_bin_to_time(np.expand_dims(test[:, 1], axis=0), window=1, fs=100, unit_overlaping=True)
ax = cplot._significance_bars(sig_start[0], sig_end[0], ax=ax)

########################################################################################################################

# attempts to take the mean difference across probes  to have a rough idea if this makes sense
probe_mean = np.mean(diff_arr, axis=0) # shape Transition x Transition x Unit x Time x Metric(mena, SEM)
plt.figure(); plt.imshow(probe_mean[1, 3, :, :, 0]) # difference continuous vs sharp for all units at all time


# attempts to take the summed significantly different bins across probes
binned_significance = np.sum(diff_arr, axis = 0 )
plt.figure(); plt.imshow(binned_significance[1, 3, :, :, 1]) # difference continuous vs sharp for all units at all time
########################################################################################################################
# for the best cell, try to collapse difference/significance across different probes, keeping identity of context
# transition
cell = cellorder.index(best_cell)
# diff_arr shape Probe x ContextTransition x ContextTransition x Units x Time x Metric

# plots significance for each type of transition (ax) plots the significance over time per individual probe (y ax)
fig, axes = plt.subplots(6, 1)
axes = np.ravel(axes)
for aa, (ct1, ct2) in enumerate(itt.combinations(context_transitions, 2)):
    ctrans1 = context_transitions.index(ct1)
    ctrans2 = context_transitions.index(ct2)
    slice = diff_arr[:, ctrans1, ctrans2, cell, :, 1]

    axes[aa].imshow(slice, aspect='auto')

# plots significance for each type of transition (color)
# plots the significance over time pooled across probes (y ax, offset)
fig, ax = plt.subplots()
for aa, (ct1, ct2) in enumerate(itt.combinations(context_transitions, 2)):
    ctrans1 = context_transitions.index(ct1)
    ctrans2 = context_transitions.index(ct2)
    slice = diff_arr[:, ctrans1, ctrans2, cell, :, 1]
    cum_sig = np.sum(slice, axis=0) #cumulative sum across
    ax.plot(cum_sig+(4*aa))

# plots significance for pooled transition plots the significance over time pooled across probes (y ax)
fig, ax = plt.subplots()
CT_pool = list()
for aa, (ct1, ct2) in enumerate(itt.combinations(context_transitions, 2)):
    ctrans1 = context_transitions.index(ct1)
    ctrans2 = context_transitions.index(ct2)
    CT_pool.append(diff_arr[:, ctrans1, ctrans2, cell, :, 1])

CT_pool = np.stack(CT_pool, axis=0)
toplot = np.sum(np.sum(CT_pool, axis=1), axis=0)  # cumulative sum across
ax.plot(toplot)

########################################################################################################################
# there  is a lot of  on reponsive cells, since we are using PSTHs and single cells, it is reasobale to do the analyss
# over the PCs

# averages away repetitios of STIM

loaded_rec['resp'] = loaded_rec['resp'].rasterize()
psth_rec = preproc.average_away_epoch_occurrences(loaded_rec, epoch_regex='^STIM_')
# defines subepochs
psth_rec = cpe.set_recording_subepochs(psth_rec, set_pairs=True)
# signal PCA
sig_pcs, stats = cpca.signal_PCA(psth_rec['resp'])
# check explained variance
fig, ax = plt.subplots()
toplot = np.cumsum(stats.explained_variance_ratio_)
ax.plot(toplot, '.-', color='black')
ax.set_xlabel('Principal Component')
ax.set_ylabel('cumulative variance explained')
ax.legend()
ax.set_title('PSTH PCA site {}'.format(site))

# check how the PCs look
# response to sound. there is something in PC3
fig, axes = cplot.hybrid(sig_pcs, epoch_names=['REFERENCE'], channels='all')
# response to the CPN triplet stimuli
epoch_names = nep.epoch_names_matching(sig_pcs.epochs, r'\ASTIM_Tsequence.*')
fig, axes = cplot.hybrid(sig_pcs, epoch_names=epoch_names, channels='all')
# plot response of all relevant probes after silence for top PCs
fig, axes = cplot.hybrid(sig_pcs, epoch_names=r'\AC0_P([679]|10)\Z', channels=[0,1,2,3,4,5])


###########################

# organizes relevnat data in array with dimentions Context x Probe x Repetition x Unit x Time
full_array, bad_cpp, good_cpp, context_names, probe_names = tp.make_full_array(sig_pcs, 'CPN')

# now calculate pairwise difference between context types
valid_probes = [6, 7, 9, 10]
context_transitions = ['silence', 'continuous', 'similar', 'sharp']
diff_arr = src.metrics.distance.pairwise_PSHT_distance(valid_probes, context_transitions, full_array, context_names, probe_names, )


###########################
# select top PC to plot
PC = 2

fig, axes = plt.subplots(6, 1)
axes = np.ravel(axes)
for aa, (ct1, ct2) in enumerate(itt.combinations(context_transitions, 2)):
    ctrans1 = context_transitions.index(ct1)
    ctrans2 = context_transitions.index(ct2)
    slice = diff_arr[:, ctrans1, ctrans2, PC, :, 0]

    axes[aa].imshow(slice, aspect='auto')

# plots difference for each type of transition (color)
# plots the difference over time pooled across probes (y ax, offset)
fig, ax = plt.subplots()
for aa, (ct1, ct2) in enumerate(itt.combinations(context_transitions, 2)):
    ctrans1 = context_transitions.index(ct1)
    ctrans2 = context_transitions.index(ct2)
    slice = diff_arr[:, ctrans1, ctrans2, PC, :, 0]
    cum_sig = np.sum(slice, axis=0) #cumulative sum across
    ax.plot(cum_sig+(4*aa))

# plots difference for pooled transition plots the significance over time pooled across probes (y ax)
fig, ax = plt.subplots()
CT_pool = list()
for aa, (ct1, ct2) in enumerate(itt.combinations(context_transitions, 2)):
    ctrans1 = context_transitions.index(ct1)
    ctrans2 = context_transitions.index(ct2)
    CT_pool.append(diff_arr[:, ctrans1, ctrans2, PC, :, 0])

CT_pool = np.stack(CT_pool, axis=0)
toplot = np.sum(np.sum(CT_pool, axis=1), axis=0)  # cumulative sum across
ax.plot(toplot)

########################################################################################################################
# population analisys as done for WIP talk 2, there is a better approach in
# scripts/190529_select_single_cells_with_high_variation.py
ctxs = [0, 1, 2, 3, 4]
prbs = [1, 2, 3 ,4]
modelname = 'resp'
site = 'AMT031a'

start = 0
end = 200
t1 = (start / 100) - 1
t2 = (end / 100) - 1

##
signal_name = '190431_{}_{}'.format(site, modelname)

func_args = {'signal': sig, 'probe_names': prbs, 'context_names': ctxs, 'shuffle_num': 1000,
             'trial_combinations': False}

shuffled_dispersion_time = cch.make_cache(function=cdisp.signal_single_trial_dispersion_pooled_shuffled,
                                          func_args=func_args,
                                          classobj_name=signal_name, recache=False,
                                          cache_folder='/home/mateo/mycache/shuffled_euclideans')
real, distribution = cch.get_cache(shuffled_dispersion_time)


fig, ax = plt.subplots()
line = real[start:end]
shade = distribution[:, start:end]
shade = np.mean(shade, axis=0) + np.std(shade, axis=0) * 2
t = np.linspace(t1, t2, len(line))
ax.plot(t, line , label='{}'.format(site), color='C0')
ax.fill_between(t, -shade , shade , alpha=0.5, color='C0')
ax.axvline(0, color='black', linestyle='--')
# ax.legend(prop={'size': 15})

ax.set_xlabel('time (s)', fontsize=18)
ax.set_ylabel('euclidean distance', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.suptitle('{}, different context minus same context for all probes'.format(site), fontsize=20 )
fig.set_size_inches(20,10)

########################################################################################################################






