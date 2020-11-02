import collections as coll
import itertools as itt

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import cpp_dispersion as cdisp
from src.data import epochs as cep, cache as ccache, cache as cch, load, reconstitute_rec as crec

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

sites = load.get_site_ids(310)



################################################
# get and reconstitute single cell recordings into population recording
pop_recs = coll.defaultdict(dict)
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
pop_sigs = coll.defaultdict(dict)
for site_key, model_recs in pop_recs.items():
    for modelname, rec in model_recs.items():
        cpp_rec = cep.set_recording_subepochs(rec, set_pairs=True)
        pop_sigs[site_key][modelname] = cpp_rec['pred'].rasterize()
    pop_sigs[site_key]['resp'] = cpp_rec['resp'].rasterize()

del(pop_sigs['BRT057b'])
del(pop_sigs['BRT037b'])


########################################################################################################################
# new significance calculation, using shuffle test across contexts
ctxs = [0, 1, 2, 3, 4]
prbs = [2]
modelname = 'resp'

site_shuffled_disp = coll.defaultdict(dict)
for site in ['BRT055b', 'BRT054b', 'BRT056b']:

    site = 'BRT056b'

    sig = pop_sigs[site][modelname]
    signal_name = '181213_{}_{}'.format(site, modelname)

    func_args = {'signal': sig, 'probe_names': prbs, 'context_names': ctxs, 'shuffle_num': 1000,
              'trial_combinations':False}

    shuffled_dispersion_time = cch.make_cache(function=cdisp.signal_single_trial_dispersion_pooled_shuffled,
                                              func_args=func_args,
                                              classobj_name=signal_name, recache=False,
                                              cache_folder='/home/mateo/mycache/shuffled_euclideans')

    real, distribution = cch.get_cache(shuffled_dispersion_time)

    site_shuffled_disp[site]['real'] = real
    site_shuffled_disp[site]['distribution'] = distribution


    # fig, ax = plt.subplots()
    # plt.figure()
    # plt.plot(np.mean(distribution, axis=0) + np.std(distribution, axis=0) * 2)
    # plt.plot(real, 'k', lw=3)
    # plt.title(site + 'C{} P{}'.format(ctxs, prbs))


start = 290
end = 400
t1 = (start / 100) - 3
t2 = (end / 100) - 3

fig, ax = plt.subplots()
y_tics = list()
y_lab = list()

for ii,(site, vals) in enumerate(site_shuffled_disp.items()):

    line = vals['real'][start:end]
    shade = vals['distribution'][:, start:end]
    shade = np.mean(shade,axis=0)+np.std(shade,axis=0)*2

    max_val = np.nanmax(np.stack([ss['real'][start:end] for ss in site_shuffled_disp.values()], axis =0))

    y_off = ii * (max_val + max_val*0.1)

    # defiens y_label positions and values
    zero = y_off
    max_y = y_off + np.nanmax(line)
    y_tics.append(zero)
    y_tics.append(max_y)
    y_lab.append(0)
    y_lab.append('{:.2f}'.format(np.nanmax(line)))

    t = np.linspace(t1, t2, len(line))
    ax.plot(t, line + y_off, label='{}'.format(site), color='C{}'.format(ii+5))
    ax.fill_between(t, -shade + y_off, shade + y_off, alpha=0.5, color='C{}'.format(ii+5))
    ax.axvline(0, color='black', linestyle='--')
    ax.legend(prop={'size': 15})
    ax.set_xlabel('time (s)', fontsize=18)
    ax.set_ylabel('euclidean distance', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)

ax.set_yticks(y_tics)
ax.set_yticklabels(y_lab)

ax.set_xlabel('time (s)', fontsize=18)
ax.set_ylabel('euclidean distance', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.suptitle('all sites, different context minus same context for all probes', fontsize=20 )
fig.set_size_inches(20,10)

fig.savefig('/home/mateo/Pictures/WIP2/181213_all_sites_diff_min_same_ctx_all_closeup.png', dpi=300)

########################################################################################################################
# all contexts one probe example

ctxs = [0, 1, 2, 3, 4]
prbs = [2]
modelname = 'resp'
site = 'BRT056b'

start = 290
end = 400
t1 = (start / 100) - 3
t2 = (end / 100) - 3

##
sig = pop_sigs[site][modelname]
signal_name = '181213_{}_{}'.format(site, modelname)

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
ax.plot(t, line , label='{}'.format(site), color='C{}'.format(ii + 5))
ax.fill_between(t, -shade , shade , alpha=0.5, color='C{}'.format(ii + 5))
ax.axvline(0, color='black', linestyle='--')
# ax.legend(prop={'size': 15})

ax.set_xlabel('time (s)', fontsize=18)
ax.set_ylabel('euclidean distance', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.suptitle('{}, different context minus same context for probe 2'.format(site), fontsize=20 )
fig.set_size_inches(20,10)

fig.savefig('/home/mateo/Pictures/WIP2/181213_{}_diff_min_same_ctx_P2_closeup.png'.format(site), dpi=300)

########################################################################################################################
# all contexts one probe example

ctxs = [0, 1, 2, 3, 4]
prbs = [1, 2, 3 ,4]
modelname = 'resp'
site = 'BRT056b'

start = 0
end = 600
t1 = (start / 100) - 3
t2 = (end / 100) - 3

##
sig = pop_sigs[site][modelname]
signal_name = '181213_{}_{}'.format(site, modelname)

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
ax.plot(t, line , label='{}'.format(site), color='C{}'.format(ii + 5))
ax.fill_between(t, -shade , shade , alpha=0.5, color='C{}'.format(ii + 5))
ax.axvline(0, color='black', linestyle='--')
# ax.legend(prop={'size': 15})

ax.set_xlabel('time (s)', fontsize=18)
ax.set_ylabel('euclidean distance', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=15)
fig.suptitle('{}, different context minus same context for all probes'.format(site), fontsize=20 )
fig.set_size_inches(20,10)

fig.savefig('/home/mateo/Pictures/WIP2/181213_{}_diff_min_same_ctx_all_P.png'.format(site), dpi=300)

