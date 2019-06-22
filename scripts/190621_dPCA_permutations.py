import matplotlib.pyplot as plt
import numpy as np
from dPCA import dPCA

import cpp_plots as cplot

import cpn_triplets as tp
from cpn_load import load
from cpn_reliability import signal_reliability

# load and format triplets from a site
site = 'AMT028b'
recs = load(site)
rec = recs['perm0']
sig = rec['resp'].rasterize()

# calculates response realiability and select only good cells to improve analysis

r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=0.1)
goodcells = goodcells.tolist()

# plots PSTHs of all probes after silence
fig, axes = cplot.hybrid(sig, epoch_names=r'\AC0_P\d\Z', channels=goodcells)

# plots PSHTs of individual best probe after all contexts
fig, axes = cplot.hybrid(sig, epoch_names=r'\AC\d_P3\Z', channels=goodcells)

# takes an example probe
full_array, invalid_cp, valid_cp, all_contexts, all_probes =tp.make_full_array(sig, channels=goodcells, smooth_window=20)


# get a specific probe after a set of different transitions

trialR = full_array[:, 1:, :, :, 100:] # get only the response to the probe and not the context

# reorders dimentions from Context x Probe x Trial x Neuron x Time  to  Trial x Neuron x Context x Probe x Time
trialR = trialR.transpose([2, 3, 0, 1, 4])
Tr, N, C, P, T = trialR.shape
# trial-average data
R = np.mean(trialR,0)
# center data
R -= np.mean(R.reshape((N,-1)),1)[:,None,None,None]

# initializes model
dpca = dPCA.dPCA(labels='cpt',regularizer='auto')
dpca.protect = ['t']

# Now fit the data (R) using the model we just instantiated. Note that we only need trial-to-trial data when we want to
# optimize over the regularization parameter.
Z = dpca.fit_transform(R,trialR)

# check for significance
significance_masks = dpca.significance_analysis(R,trialR,axis='t',n_shuffles=10,n_splits=10,n_consecutive=10)

# plots the first PC projection of each context, for each marginalization
# includes a measurement of significance by shuffle test (?)
time = np.arange(T)
fig, axes = plt.subplots(1, len(Z))
axes = np.ravel(axes)

for vv, (var_name, arr) in enumerate(Z.items()):

    for c in range(C):
        axes[vv].plot(time, arr[0,c,:])#, label=transitions[c])
    axes[vv].set_title(f'1st {var_name} component')

    # if var_name in significance_masks:
    #
    #     left, right = axes[vv].get_xlim()
    #     bottom, top = axes[vv].get_ylim()
    #     Ychunk = (top-bottom)/10
    #     axes[vv].set_ylim(bottom-Ychunk, top)
    #     axes[vv].imshow(significance_masks[var_name][0][None,:],
    #                     extent=[0, T, bottom-Ychunk, bottom], aspect='auto')

    if vv == len(Z)-1:
        axes[vv].legend()

# fig.suptitle(f'probe {probe}')