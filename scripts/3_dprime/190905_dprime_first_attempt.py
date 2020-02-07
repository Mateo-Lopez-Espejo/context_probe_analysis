import matplotlib.pyplot as plt
import numpy as np
from progressbar import ProgressBar
from cpn_load import load
from reliability import signal_reliability
import cpn_dprime as cpd
import cpn_dPCA as cdPCA
import fancy_plots as cplt

"""
Previous iterations on the euclidean distance analysis were unadecuate to determine whether population codes helped 
discrimination of contexts when compared with single cell representations.

Here I am using a Discrimination index (d prime, d') suggested by charlie to compare single cell vs population 
discrimination

The population discriminations is defined over the projection to a one dimentional axis (selected through LDA or dPCA)

The single cell discriminations is defined as the n-dimesional pythagorean hypotenuse of discrimination across every 
single cell dimension.


19-09-24 Update

In a second approach to the uncorrelated cells (the current paradimg being used by charlie and Stephene), I parametrically
simulate single neuron response following a normal distribution (mean and standard error calculated independently for 
each neuron). The rational behind this simulation lies in the independent parameteric simulation of neuronal responses 
which eliminates any systematic noise correlation between neurons.

d'prime was calculated then over the projection of the simulated responses into the top contexte marginalization 
PC previously defined with the dPCA over the real data. 

this was done a thousand times to define a nonparametric test of significant difference between the real d' and the 
simulated d'

The final plots show tow example sites, with the real d'prime and an example simulated d' or the SEM CI of the all the
sims. Finally it also shows vertical bars on the time bins where there is significance difference (p<0.05) between the
real d' and the sims 

End word:
the stable discriminator (single dPCA projection across time) makes some sense as it would simplify the job for a 
downstream decoder, however there is no reason for not having a felxible discriminator, thus calculating an LDA for each
time bin is justifiable, furthermore it would give us an estimate of the best case scenario of maximum discrimination

The gaussian parametric simulation might not be the best option, a better approache would use poisson firing. 

the inner workins and meaning of the dPCA marginalziations still eludes us
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

    # gets the trial-wise projection in the top PC of the context marginalization for a pairs of contexts
    # calculates population (one dimensional) d' for a pair of contexts
    real_proj_ctx0 = trialZ['ct'][:, 0, c0, :]
    real_proj_ctx1 = trialZ['ct'][:, 0, c1, :]
    real_dprime = cpd.dprime(real_proj_ctx0, real_proj_ctx1,absolute=True)


    # gets the real data raster (no dim reduction) to calculate single cell, population independent d'
    raster = cdPCA.raster_from_sig(sig,probe,channels=goodcells, transitions=meta['transitions'],
                                       smooth_window=meta['smoothing_window'],raster_fs=meta['raster_fs'])

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, centers = cdPCA.format_raster(raster)
    trialR, R = trialR.squeeze(), R.squeeze()


    ctx0 = trialR[:, :, c0, :]
    ctx1 = trialR[:, :, c1, :]

    # parametrically simulates new data (asumes gausian distribution) trialSimulation -> trialS
    # then transforms into the dPCA marginalizations previously defined withe the real data
    # does this n times to generate a distribution...

    nreps = 10000
    all_sims = np.zeros([real_dprime.shape[0], nreps]) # shape Time x Repetitions
    pbar = ProgressBar()
    for rep in pbar(range(nreps)):
        # normal simulation
        trialS = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0), size=trialR.shape)
        trialS = trialS.squeeze()
        trialSR = cdPCA.transform_trials(dpca, trialS)

        # calculates the d' with the new simulated data.
        sim_proj_ctx0 = trialSR['ct'][:, 0, c0, :]
        sim_proj_ctx1 = trialSR['ct'][:, 0, c1, :]
        sim_dprime = cpd.dprime(sim_proj_ctx0, sim_proj_ctx1,absolute=True)
        all_sims[:, rep] = sim_dprime


    # calculates the one tailed pvalue of the difference
    pvalue = np.sum((all_sims.T > real_dprime), axis=0) / nreps
    sig_times = cplt._sig_bin_to_time(pvalue<0.05, 1, 100,)


    # # calculates single cell (n dimensional) d' as the pythagorean of each dimension d'
    # ndim_dprime = cpd.ndim_dprime(ctx0, ctx1, absolute=True)

    # plots the different variants of d'
    fig, axes = plt.subplots(1,2)
    axes = np.ravel(axes)
    t = np.linspace(0, 1, 100, False)
    axes[0].plot(t, real_dprime, color='black', label='real dprime')
    axes[0].plot(t, np.mean(all_sims,axis=1), color='green', label='sim dprime')
    axes[0].set_title('example simulation')
    axes[0].legend()
    cplt._significance_bars(sig_times[0], sig_times[1], ax=axes[0])


    # plots real d' and shadow of shuffled dprimes
    cplt.plot_dist_with_CI(real_dprime, [all_sims.T], ['param_sim\nconfidence interval'], ['green'], 0, 100, 0, 100, ax=axes[1], show_labels=True)
    axes[1].legend()
    axes[1].set_title(f'{nreps} reps MSE confidence interval')
    cplt._significance_bars(sig_times[0], sig_times[1], ax=axes[1])


    fig.suptitle(f"{site}, probe {probe}; {meta['transitions'][c0]} vs {meta['transitions'][c1]}")