import collections as col
import itertools as itt

import cpn_load
import cpp_cache as ccache
import cpp_dispersion as cdisp
import cpp_epochs as cep
import cpp_reconstitute_rec as crec
import fancy_plots as cplot

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sst

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



################################################
# get and reconstitute single cell recordings into population recording
pop_recs = col.defaultdict(dict)
for (site_name, cells), modelname in itt.product(sites.items(), all_models):
    print('#####\nreconstituting site {} with model {}\n '.format(site_name, modelname))
    recons_args = {'batch':310, 'cellid_list':cells, 'modelname': modelname}
    recons_cache = ccache.make_cache(crec.reconsitute_rec, func_args=recons_args, classobj_name='reconstitution',
                                     recache=False, cache_folder='/home/mateo/mycache/reconstitute_recs',
                                     use_hash=True)
    reconstituted_recording = ccache.get_cache(recons_cache)
    pop_recs[site_name][modelname] = reconstituted_recording

# if pop_recs['BRT057b']:
#     del(pop_recs['BRT057b'])

################################################
# reorders in dictionary of signals, including only the response and the prediction of each mode
# reformats the epochs
pop_sigs = col.defaultdict(dict)
for site_key, model_recs in pop_recs.items():
    for modelname, rec in model_recs.items():
        cpp_rec = cep.set_recording_subepochs(rec, set_pairs=True)
        pop_sigs[site_key][modelname] = cpp_rec['pred'].rasterize()
    pop_sigs[site_key]['resp'] = cpp_rec['resp'].rasterize()

del(pop_sigs['BRT057b'])


########################################################################################################################
# cleaner version of example for wip talk.... something broke this plotting function down the road, not on my code

best_site = 'BRT056b'
best_cell = 'BRT056b-58-1'

sig = pop_sigs[best_site]['resp']
fig, axes = cplot.hybrid(sig, epoch_names=r'\AC\d_P3', start=3, end=6, scatter_kws={'alpha': 0.1})
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.tight_layout()
for ax in axes: ax.axis('off')
fig.suptitle('')
# fig.suptitle('stim_num independent vocalizations PSTH\nsite {}'.format(best_site))
# fig.savefig('/home/mateo/Pictures/WIP2/181207_eg_cpp_all_cells_{}.png'.format(best_site), dpi=100)
# fig.savefig('/home/mateo/Pictures/WIP2/181207_eg_cpp_all_cells_{}.svg'.format(best_site))


# single cell, one prb, all contexts
goodprobe = 3
goodcell = 'BRT056b-58-1'
fig, axes = cplot.hybrid(sig, epoch_names=r'\AC\d_P{}'.format(goodprobe), channels=[goodcell],
             start=3, end=6,scatter_kws={'alpha': 0.2})
ax = axes[0]
ax.set_xlabel('seconds')
fig.suptitle('stim_num dependent response to prb {}'.format(goodprobe))
fig.set_size_inches(13,9)
# fig.savefig('/home/mateo/Pictures/WIP2/181205_CPP_probe{}_PSTH_cell_{}.png'.format(goodprobe, goodcell), dpi=200)
# fig.savefig('/home/mateo/Pictures/DAC1/181205_CPP_probe{}_PSTH_cell_{}.svg'.format(ii, best_cell))



# single cell, one prb, all contexts, close up in time
goodprobe = 3
goodcell = 'BRT056b-58-1'
fig, axes = cplot.hybrid(sig, epoch_names=r'\AC\d_P{}'.format(goodprobe), channels=[goodcell],
             start=0, end=None,scatter_kws={'alpha': 0.2}, time_strech=[2.99, 3.5], time_offset=-0.01)
for ax in axes: ax.axis('off'); ax.set_title(''); ax.axvline(0,color='black')

fig.suptitle('')
fig.set_size_inches(13,9)
plt.tight_layout()
fig.savefig('/home/mateo/Pictures/WIP2/181213_CPP_probe{}_PSTH_cell_{}_closeup.png'.format(goodprobe, goodcell), dpi=200)

# single cell, one prb, all contexts, close up in time. Similar as previous plot but with only two overlayed examples

examples = sig.extract_epochs(['C4_P3', 'C3_P3'])
cell = 10
fig, ax = plt.subplots()
for cpp, arr in examples.items():
    color = 'C{}'.format(cpp[1])
    strech = arr[:, cell, 298:350]
    mean = np.mean(strech, axis=0)
    sem = sst.sem(strech, axis=0)
    T = np.linspace(-0.02, 0.5, strech.shape[-1])

    ax.plot(T, mean, color=color)
    ax.fill_between(T, mean-sem, mean+sem, color=color, alpha=0.4)

ax.axis('off'); ax.set_title(''); ax.axvline(0,color='black')

fig.suptitle('')
fig.set_size_inches(13,9)
plt.tight_layout()

fig.savefig('/home/mateo/Pictures/WIP2/181213_CPP_probe{}_PSTH_cell_{}_pair_comparison.png'.format(goodprobe, goodcell), dpi=200)
















