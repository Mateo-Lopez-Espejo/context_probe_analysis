from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl
import nems_lbhb.baphy_io as io
from mpl_toolkits.axes_grid.inset_locator import InsetPosition
from src.metrics.dprime import ndarray_dprime

parmfile1 = '/auto/data/daq/Teonancatl/TNC006/TNC006a09_p_NON.m'
parmfile2 = '/auto/data/daq/Teonancatl/TNC006/TNC006a10_p_NON.m'

animal = 'Teonancatl'

rasterfs = 1000
recache = False
options = {'resp': True, 'rasterfs': rasterfs, 'stim':False}
manager = BAPHYExperiment(parmfile=[parmfile1, parmfile2])
tstart = -0.02
tend = 0.1


rec = manager.get_recording(recache=recache, **options)
rec['resp'] = rec['resp'].rasterize()
prestim = rec['resp'].extract_epoch('PreStimSilence').shape[-1] / rasterfs
m = rec.copy().and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
poststim = (rec['resp'].extract_epoch('REFERENCE', mask=m['mask'], allow_incomplete=True).shape[-1] / rasterfs) + prestim
lim = (-prestim, tend )
lim = (tstart, tend)
s = 1


# get light on / off
opt_data = rec['resp'].epoch_to_signal('LIGHTON')
opt_ref_mask = opt_data.extract_epoch('REFERENCE').any(axis=(1, 2))

opt_s_stop = (np.argwhere(np.diff(opt_data.extract_epoch('REFERENCE')[opt_ref_mask, :, :][0].squeeze())) + 1) / rasterfs

# Organizes data in an array with shape Optic_stim(off,on) x Trials x Neurons x Time
references = rec['resp'].extract_epoch('REFERENCE')
ref_light_on = references[opt_ref_mask,:,:]
ref_light_off = references[np.logical_not(opt_ref_mask),:,:]

# same with just the light response part
stim = rec['resp'].extract_epoch('Stim')
stim_light_on = stim[opt_ref_mask,:,:]
stim_light_off = stim[np.logical_not(opt_ref_mask),:,:]


# determines difference between light on and light off trials and sorts neuron by max max effect
opt_stim_dprime  = ndarray_dprime(stim_light_on, stim_light_off, axis=0)

best_neuron_idx = np.flip(np.argsort(np.max(np.abs(opt_stim_dprime), axis=1)))

neurons = np.asarray(rec['resp'].chans)
sorted_neurons = np.asarray(neurons)[best_neuron_idx]
#
# fig, ax = plt.subplots()
# ax.plot(opt_stim_dprime[best_neuron_idx[0],:])

savefig = False
DIR = pl.Path('')


def opto_plot(neuron, axes=(None, None)):

    if axes[0] == None:
        fig, axes = plt.subplots(2,1, squeeze=True)

    r = rec['resp'].extract_channels([neuron]).extract_epoch('REFERENCE').squeeze()

    # psth
    on = r[opt_ref_mask, :].mean(axis=0) * options['rasterfs']
    on_sem = r[opt_ref_mask, :].std(axis=0) / np.sqrt(opt_ref_mask.sum()) * options['rasterfs']
    t = np.arange(0, on.shape[-1] / options['rasterfs'], 1 / options['rasterfs']) - prestim
    axes[1].plot(t, on, color='blue')
    axes[1].fill_between(t, on - on_sem, on + on_sem, alpha=0.3, lw=0, color='blue')
    off = r[~opt_ref_mask, :].mean(axis=0) * options['rasterfs']
    off_sem = r[~opt_ref_mask, :].std(axis=0) / np.sqrt((~opt_ref_mask).sum()) * options['rasterfs']
    t = np.arange(0, off.shape[-1] / options['rasterfs'], 1 / options['rasterfs']) - prestim
    axes[1].plot(t, off, color='grey')
    axes[1].fill_between(t, off - off_sem, off + off_sem, alpha=0.3, lw=0, color='grey')
    axes[1].set_ylabel('Spk / sec')
    axes[1].set_xlim(lim[0], lim[1])

    # spike raster / light onset/offset
    st = np.where(r[opt_ref_mask, :])
    axes[0].scatter((st[1] / rasterfs) - prestim, st[0], s=s, color='b')
    offset = st[0].max()
    st = np.where(r[~opt_ref_mask, :])
    axes[0].scatter((st[1] / rasterfs) - prestim, st[0] + offset, s=s, color='grey')
    for ss in opt_s_stop:
        axes[0].axvline(ss - prestim, linestyle='--', color='lime')
    axes[0].set_title(neuron)
    axes[0].set_ylabel('Rep')
    axes[0].set_xlim(lim[0], lim[1])

    # add inset for mwf
    axes[1] = plt.axes([axes[1].get_subplotspec().colspan.start, axes[1].get_subplotspec().colspan.start,
                    axes[1].get_subplotspec().rowspan.start, axes[1].get_subplotspec().rowspan.start])

    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(axes[1], [0.5, 0.5, 0.5, 0.5])
    axes[1].set_axes_locator(ip)
    mwf = io.get_mean_spike_waveform(str(neuron), animal, usespkfile=True)
    axes[1].plot(mwf, color='red')
    axes[1].axis('off')
    axes[1].set_xlabel('Time from light onset (sec)')

    return fig, axes

plt.ioff()
good_neurons = []
for neuron in sorted_neurons:
    plt.close('all')
    fig, axes = opto_plot(neuron)
    plt.show()
