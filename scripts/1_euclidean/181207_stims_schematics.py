import collections as coll
import itertools as itt

import cpn_load
import cpp_cache as ccache
import cpp_dispersion as cdisp
import cpp_epochs as cep
import cpp_reconstitute_rec as crec
import cpp_PCA as cpca
import fancy_plots as cplt
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.signal as ssig
import scipy.ndimage.filters as sf

import nems.epoch as nep

batch = 310

all_models = ['wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1', 'wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1',
              'wc.2x2.c-fir.2x15-lvl.1-stategain.S-dexp.1', 'wc.2x2.c-fir.2x15-lvl.1-dexp.1']

shortnames = {'resp': 'resp',
              'wc.2x2.c-fir.2x15-lvl.1-dexp.1': 'LN',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1': 'STP',
              'wc.2x2.c-fir.2x15-lvl.1-stategain.S-dexp.1': 'state',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1': 'STP_state'}

color_dict = {'resp': 'C0',
              'wc.2x2.c-fir.2x15-lvl.1-dexp.1': 'C1',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1': 'C2',
              'wc.2x2.c-fir.2x15-lvl.1-stategain.S-dexp.1': 'C3',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1': 'C4'}

voc_color = {'voc_{}'.format(cc): 'C{}'.format(cc) for cc in range(5)}
voc_cmpat = {'voc_0': 'Blues', 'voc_1': 'Oranges', 'voc_2': 'Greens', 'voc_3': 'Reds', 'voc_4': 'Purples'}
sites = cpn_load.get_site_ids(310)


# selects the best site and a the best model to ease procesing
best_site = 'BRT056b'
best_model = 'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1'
cells = sites[best_site]

# reconstitutes population recording
print('#####\nreconstituting site {} with model {}\n '.format(best_site, best_model))
recons_args = {'batch': 310, 'cellid_list': cells, 'modelname': best_model}
recons_cache = ccache.make_cache(crec.reconsitute_rec, func_args=recons_args, classobj_name='reconstitution',
                                 recache=False, cache_folder='/home/mateo/mycache/reconstitute_recs',
                                 use_hash=True)
rrec = ccache.get_cache(recons_cache)

# formats with CPP epochs and gets
rec = cep.set_recording_subepochs(rrec, set_pairs=False)

# get the envelopes
all_stims = rec['stim'].rasterize().extract_epochs(['voc_1', 'voc_2', 'voc_3', 'voc_4'])

# gets an example of each stimulus for both channels
all_stims = {stim_name: arr[0,:,:] for stim_name, arr in all_stims.items()}

envelopes = {'env_0': all_stims['voc_1'][0,:],
             'env_1': all_stims['voc_3'][0,:]}

# normalzies and resamples envelopes for resolution purporse
normalized = {key: val / np.max(val) for key, val in envelopes.items()}
resampled  = {key: ssig.resample(arr,3000) for key, arr in normalized.items()}


# generates sine waves of the right length

Fs = 8000
sample = 3000

f0 = 80
f1 = 150
x = np.arange(sample)
carrier = dict()
carrier['car_0'] = np.sin(2 * np.pi * f0 * x / Fs)
carrier['car_1'] = np.sin(2 * np.pi * f1 * x / Fs)

#  for each envelope, generates a version of the sinewave scaled to the envelope
new_stims = dict()
fig, axes = plt.subplots(3,3)
for nn, ((ee, env), (cc, carr)) in enumerate(itt.product(resampled.items(), carrier.items())):
    col = int(np.floor(nn/2))+1
    row = (nn%2)+1

    if row == 1:
        ax = axes[row-1, col]
        ax.plot(env, color='black')
    if col == 1:
        ax = axes[row, col-1]
        ax.plot(carr, color='black')

    ax = axes[row, col]
    nstim = env * carr
    nstim = nstim / np.max(nstim)
    ax.plot(nstim, color='C{}'.format(nn+1))
    ax.plot(env, color='black')

    stim_name = 'voc_{}'.format(nn +1 )
    new_stims[stim_name] = nstim

for ax in np.ravel(axes): ax.axis('off')

plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
fig.set_size_inches(15,10)
fig.savefig('/home/mateo/Pictures/WIP2/181207_stim_construction.png', dpi=300)
# fig.savefig('/home/mateo/Pictures/WIP2/181207_stim_construction.svg')


# generates the sequence of stimuli

ep_names = rec.epochs.name.unique()
order = {name[15]: [int(nn) for nn in name.split(':')[-1].split('  ')]
            for name in ep_names if name.startswith('STIM')}



#  plots the trains of stimuli one over the other
stim_len = new_stims['voc_1'].shape[0]
silence = np.zeros(stim_len)
new_stims['voc_0'] = silence
fig = plt.figure(figsize=(8, 10))
ax1 = fig.add_axes([0, 0.5, 1, 0.5])
width = (1/7)*2
left = 0.5- width/2
ax2 = fig.add_axes([left,0,width,0.5])

v_offset = 2
eg_probe = 'voc_1'


for tt in range (1, 5):
    train = order[str(tt)]
    v_off = v_offset * (tt + 1)
    ax1.plot(silence + v_off, color='C0')
    for vv, voc_num in enumerate(train):
        h_offset = (vv + 1) * stim_len
        voc = 'voc_{}'.format(voc_num)
        xx = np.arange(h_offset, h_offset + stim_len)
        ax1.plot(xx, new_stims[voc] + v_off, color=voc_color[voc])
    ax1.plot(xx+stim_len, silence +  v_off, color='C0')

ax1.axis('off')

# plots an example in which all the instnaces of a prb are ploted with the preceding stim_num
for cpp, (context, wave) in enumerate(new_stims.items()):
    # if stim_num == eg_probe:
    #     continue
    v_off = v_offset * cpp
    xx = np.arange(stim_len, stim_len*2)
    label = 'stim_num {}, prb {}'.format(context.split('_')[-1], eg_probe.split('_')[-1])
    ax2.plot(wave - v_off, color=voc_color[context], label=label)
    ax2.plot(xx, new_stims[eg_probe] - v_off, color=voc_color[eg_probe])


ax2.axis('off')

fig.savefig('/home/mateo/Pictures/WIP2/181207_cpp_slicing.png', dpi=300)
# fig.savefig('/home/mateo/Pictures/WIP2/181207_cpp_slicing.svg')

## simple example of 13 dimentional response to show raw response to stimulus

rec = cep.set_recording_subepochs(rrec, set_pairs=True)
sig = rec['resp'].rasterize()

arr = sig.extract_epoch('C2_P1')
arr = np.mean(arr, axis=0)
first = arr[:,0:300]
second = arr[:,300:]

time_first = np.linspace(-3, 0, 300)
time_second = np.linspace(0, 3, 300)

fig, ax = plt.subplots()
ticks = list()
for ii in range(arr.shape[0]):
    y_off = ii * 1.5
    ticks.append(y_off)
    ax.plot(time_first, first[ii,:].T + y_off, color='C2')
    ax.plot(time_second, second[ii,:].T + y_off, color='C1')

ax.set_yticks(ticks)
ax.set_yticklabels(['cell {}'.format(ii) for ii in range(len(ticks))])
plt.tick_params(axis='both', which='major', labelsize=15)

# ax.axis('off')
fig.set_size_inches(10,10)
plt.tight_layout()
for ss, spine in enumerate(plt.gca().spines.values()):
    spine.set_visible(False)

fig.savefig('/home/mateo/Pictures/WIP2/181209_example_firing_rate.png', dpi=300)



