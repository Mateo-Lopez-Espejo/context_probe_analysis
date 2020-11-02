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
################################################
# iterates over each site id, and calculates the dispersion for predictions of each different models
# Note: this is an old approahc, see the next code block

site_dispersions = coll.defaultdict(lambda: coll.defaultdict(dict))
for site, model_sig in pop_sigs.items():
    # for multi core running
    # site_IDs = list(pop_recs.keys())
    # site = site_IDs[4]
    # recs = pop_recs[site]
    # calculatest the dispersion for each prediction and one response
    if site == 'BRT037b':
        probe_name = [1,2,3,4]
    else:
        probe_name = [0,1,2,3,4]

    for sss, shuffle in zip(['real', 'shuffle'],[False, True]):

        for modelname, sig in model_sig.items():
            site_dispersions[site][sss][modelname] = cdisp.signal_single_trial_dispersion_digest(
                sig, probe_names=probe_name, context_names=[0, 1, 2, 3, 4], shuffle_neurons=shuffle,
                signal_name='{}_{}'.format(site, modelname), recache=True)

################################################
# single run to debug
'''
probes = [0,1,2,3,4]
debuged = dict()
site = 'BRT056b'
modelname = 'resp'
sig = pop_sigs[site][modelname]
for sss, shuffle in zip(['real', 'shuffle'],[False, True]):
     debuged[sss] = cdisp.signal_single_trial_dispersion_digest(
         sig, probe_names=probes, context_names=[0, 1, 2, 3, 4], shuffle_neurons=shuffle,
         signal_name='{}_{}'.format(site, modelname), recache=True)
'''

########################################################################################################################
# pooling of contexts pairs by category, for example site and actual response
site = 'BRT056b'
modelname = 'resp'
sig = pop_sigs[site][modelname]
ctxs = prbs = [0,1,2,3,4]
pooled_dict = cdisp.signal_single_trial_dispersion_pooled(signal=sig, probe_names=ctxs, context_names=prbs,
                                                          shuffle_neurons=False, signal_name='{}_{}'.format(site, modelname),
                                                          recache=False)

# rand_pooled_dict = cdisp.signal_single_trial_dispersion_pooled(signal=sig, probe_names=ctxs, context_names=prbs,
#                                                           shuffle_neurons=True, signal_name='{}_{}'.format(site, modelname),
#                                                           recache=False)

all_data = {'real': pooled_dict,}# 'shuffled': rand_pooled_dict}


data_set = ['real', 'real']
probe = 'P_all'
# ['all_ctx', 'same_ctx', 'diff_ctx', 'diff_car', 'diff_env', 'diff_all', 'none_ctx']
pools = ['diff_ctx', 'same_ctx' ]
colors =['blue', 'red']

fig, axes = plt.subplots(1,2, sharex=True)
axes = np.ravel(axes)
for ds, pp, cc in zip(data_set, pools, colors):

    # plots the mean as a line
    mean = all_data[ds][probe][pp]['mean']
    t = np.linspace(-3,3, len(mean))

    axes[0].plot(t, mean, color=cc, label='{} {}'.format(ds, pp))

    # plots the sem as a shaded area
    sem = all_data[ds][probe][pp]['sem']
    axes[0].fill_between(t, mean + sem, mean - sem, facecolor=cc, alpha=0.5)
axes[0].legend()

pool0 = all_data[data_set[0]][probe][pools[0]]['mean']
pool1 = all_data[data_set[1]][probe][pools[1]]['mean']
sem0 = all_data[data_set[0]][probe][pools[0]]['sem']
sem1 = all_data[data_set[1]][probe][pools[1]]['sem']

diff =  pool0 - pool1
sem_sum = sem0+sem1
sig_diff = np.abs(diff) > sem_sum

t = np.linspace(-3,3, len(diff))
axes[1].plot(t, diff, label='{} {} minus {} {}'.format(data_set[0], pools[0], data_set[1], pools[1]), color='black')
axes[1].fill_between(t, -sem_sum, sem_sum, alpha=0.5, color='gray')
axes[1].legend()
fig.suptitle('site {}, {}, probe {}'.format(site, modelname, probe))





########################################################################################################################
# different probes, pooling of contexts pairs by category.
site = 'BRT056b'
modelname = 'resp'
sig = pop_sigs[site][modelname]
ctxs = prbs = [0,1,2,3,4]
pooled_dict = cdisp.signal_single_trial_dispersion_pooled(signal=sig, probe_names=ctxs, context_names=prbs,
                                                          shuffle_neurons=False, signal_name='{}_{}'.format(site, modelname),
                                                          recache=False)
pairs_pooled_dict = cdisp.signal_single_trial_dispersion_diff_probe_pooled(signal=sig, probe_names=ctxs, context_names=prbs,
                                                          shuffle_neurons=False, signal_name='{}_{}'.format(site, modelname),
                                                          recache=False)

# PSTH_pooled_dict = cdisp.signal_single_trial_dispersion_pooled(signal=sig, probe_names=ctxs, context_names=prbs,
#                                                           shuffle_neurons='PSTH', signal_name='{}_{}'.format(site, modelname),
#                                                           recache=False)

# rand_pooled_dict = cdisp.signal_single_trial_dispersion_pooled(signal=sig, probe_names=ctxs, context_names=prbs,
#                                                           shuffle_neurons=True, signal_name='{}_{}'.format(site, modelname),
#                                                           recache=False)

all_data = {'real': {**pooled_dict, **pairs_pooled_dict}}# 'shuffled': rand_pooled_dict}

data_set = ['real', 'real']
probe = 'PvP_all'
# ['all_ctx', 'same_ctx', 'diff_ctx', 'diff_car', 'diff_env', 'diff_all', 'none_ctx']
pools = ['all_ctx', 'same_ctx']
colors =['blue', 'red']

fig, axes = plt.subplots(1,2, sharex=True)
axes = np.ravel(axes)
for ds, pp, cc in zip(data_set, pools, colors):

    # plots the mean as a line
    mean = all_data[ds][probe][pp]['mean']
    t = np.linspace(-3,3, len(mean))

    axes[0].plot(t, mean, color=cc, label='{} {}'.format(ds, pp))

    # plots the sem as a shaded area
    sem = all_data[ds][probe][pp]['sem']
    axes[0].fill_between(t, mean + sem, mean - sem, facecolor=cc, alpha=0.5)
axes[0].legend()

pool0 = all_data[data_set[0]][probe][pools[0]]['mean']
pool1 = all_data[data_set[1]][probe][pools[1]]['mean']
sem0 = all_data[data_set[0]][probe][pools[0]]['sem']
sem1 = all_data[data_set[1]][probe][pools[1]]['sem']

diff =  pool0 - pool1
sem_sum = sem0+sem1
sig_diff = np.abs(diff) > sem_sum

t = np.linspace(-3,3, len(diff))
axes[1].plot(t, diff, label='{} {} minus {} {}'.format(data_set[0], pools[0], data_set[1], pools[1]), color='black')
axes[1].fill_between(t, -sem_sum, sem_sum, alpha=0.5, color='gray')
axes[1].legend()
fig.suptitle('site {}, {}, probe {}'.format(site, modelname, probe))

########################################################################################################################
# good examples, first single
site = 'BRT056b'
modelname = 'resp'
sig = pop_sigs[site][modelname]
ctxs = prbs = [0,1,2,3,4]
means, SEMs, names = cdisp.signal_single_trial_dispersion_digest(signal=sig, probe_names=ctxs, context_names=prbs,
                                                          shuffle_neurons=False, signal_name='{}_{}'.format(site, modelname),
                                                          recache=False)

probe = 'P2'
context_pairs = ['C1_vs_C4', 'C1_vs_C1']
colors = ['green', 'black']

m_dict = dict()
s_dict = dict()

fig, axes = plt.subplots(1,2, sharex=True)
axes = np.ravel(axes)

for cp, cc in zip(context_pairs, colors):
    idx = names[probe].index(cp)
    mean = means[probe][idx, :]
    m_dict[cp] = mean

    SEM = SEMs[probe][idx, :]
    s_dict[cp] = SEM
    t = np.linspace(-3, 3, len(mean))

    axes[0].plot(t, mean, color=cc, label='{}'.format(cp))
    axes[0].fill_between(t, mean + SEM, mean - SEM, facecolor=cc, alpha=0.5)
    axes[0].axvline(0, color='black', linestyle='--')
axes[0].legend(prop={'size': 15})
axes[0].set_xlabel('time (s)', fontsize=18)
axes[0].set_ylabel('euclidean distance', fontsize=18)
axes[0].tick_params(axis='both', which='major', labelsize=15)

diff =  m_dict[context_pairs[0]] - m_dict[context_pairs[1]]
sem_sum = s_dict[context_pairs[0]] - s_dict[context_pairs[1]]
sig_diff = np.abs(diff) > sem_sum

t = np.linspace(-3,3, len(diff))
axes[1].plot(t, diff, label='{} minus {}'.format(context_pairs[0], context_pairs[1]), color='black')
axes[1].fill_between(t, -sem_sum, sem_sum, alpha=0.5, color='gray')
axes[1].axvline(0, color='black', linestyle='--')
axes[1].legend(prop={'size': 15})
axes[1].set_xlabel('time (s)', fontsize=18)
axes[1].set_ylabel('euclidean distance', fontsize=18)
axes[1].tick_params(axis='both', which='major', labelsize=15)

fig.suptitle('site {}, probe {}'.format(site, probe), fontsize=20)
fig.set_size_inches(20,10)
fig.savefig('/home/mateo/Pictures/WIP2/181212_eg_{}_{}_{}_vs_{}.png'.format(site, site, context_pairs[0], context_pairs[1]), dpi=300)


########################################################################################################################
# example probe, pooling all contexts pairs
# ['BRT055b', 'BRT054b', 'BRT056b']

site = 'BRT056b'
probe = 'P2'

for probe, site in itt.product(['P2', 'P_all'], ['BRT055b', 'BRT054b', 'BRT056b']):
    if probe == 'P2':
        if site == 'BRT056b':
            pass
        else:
            continue

    modelname = 'resp'
    sig = pop_sigs[site][modelname]
    ctxs = prbs = [0,1,2,3,4]
    pooled_dict = cdisp.signal_single_trial_dispersion_pooled(signal=sig, probe_names=ctxs, context_names=prbs,
                                                              shuffle_neurons=False, signal_name='{}_{}'.format(site, modelname),
                                                              recache=False)


    all_data = {'real': pooled_dict}# 'shuffled': rand_pooled_dict}

    data_set = ['real', 'real']
    # ['all_ctx', 'same_ctx', 'diff_ctx', 'diff_car', 'diff_env', 'diff_all', 'none_ctx']
    pools = ['diff_ctx', 'same_ctx']
    colors =['green', 'black']

    fig, axes = plt.subplots(1,2, sharex=True)
    axes = np.ravel(axes)
    for ds, pp, cc in zip(data_set, pools, colors):

        # plots the mean as a line
        mean = all_data[ds][probe][pp]['mean']
        t = np.linspace(-3,3, len(mean))

        axes[0].plot(t, mean, color=cc, label='{}'.format(pp))

        # plots the sem as a shaded area
        sem = all_data[ds][probe][pp]['sem']
        axes[0].fill_between(t, mean + sem, mean - sem, facecolor=cc, alpha=0.5)
        axes[0].axvline(0, color='black', linestyle='--')
    axes[0].legend(prop={'size': 15})
    axes[0].set_xlabel('time (s)', fontsize=18)
    axes[0].set_ylabel('euclidean distance', fontsize=18)
    axes[0].tick_params(axis='both', which='major', labelsize=15)

    pool0 = all_data[data_set[0]][probe][pools[0]]['mean']
    pool1 = all_data[data_set[1]][probe][pools[1]]['mean']
    sem0 = all_data[data_set[0]][probe][pools[0]]['sem']
    sem1 = all_data[data_set[1]][probe][pools[1]]['sem']

    diff =  pool0 - pool1
    sem_sum = sem0+sem1
    sig_diff = np.abs(diff) > sem_sum

    t = np.linspace(-3,3, len(diff))
    axes[1].plot(t, diff, label='{} minus {}'.format(pools[0], pools[1]), color='black')
    axes[1].fill_between(t, -sem_sum, sem_sum, alpha=0.5, color='gray')
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].legend(prop={'size': 15})
    axes[1].set_xlabel('time (s)', fontsize=18)
    axes[1].set_ylabel('euclidean distance', fontsize=18)
    axes[1].tick_params(axis='both', which='major', labelsize=15)
    fig.suptitle('site {}, {}, probe {}'.format(site, modelname, probe))

    fig.suptitle('site {}, probe {}'.format(site, probe), fontsize=20)
    fig.set_size_inches(20,10)

    # fig.savefig('/home/mateo/Pictures/WIP2/181212_eg_{}_{}_{}_vs_{}.png'.format(site, probe, pools[0], pools[1]), dpi=300)

    ########################################################################################################################
    # example probe, pooling all contexts pairs (like previous) close up to one second

    start = 290
    end = 600
    t1 = (start/100) - 3
    t2 = (end/100) - 3

    fig, axes = plt.subplots(1,2, sharex=True)
    axes = np.ravel(axes)
    for ds, pp, cc in zip(data_set, pools, colors):

        # plots the mean as a line
        mean = all_data[ds][probe][pp]['mean'][start:end]
        t = np.linspace(t1,t2, len(mean))

        axes[0].plot(t, mean, color=cc, label='{}'.format(pp))

        # plots the sem as a shaded area
        sem = all_data[ds][probe][pp]['sem'][start:end]
        axes[0].fill_between(t, mean + sem, mean - sem, facecolor=cc, alpha=0.5)
        axes[0].axvline(0, color='black', linestyle='--')
    axes[0].legend(prop={'size': 15})
    axes[0].set_xlabel('time (s)', fontsize=18)
    axes[0].set_ylabel('euclidean distance', fontsize=18)
    axes[0].tick_params(axis='both', which='major', labelsize=15)

    pool0 = all_data[data_set[0]][probe][pools[0]]['mean'][start:end]
    pool1 = all_data[data_set[1]][probe][pools[1]]['mean'][start:end]
    sem0 = all_data[data_set[0]][probe][pools[0]]['sem'][start:end]
    sem1 = all_data[data_set[1]][probe][pools[1]]['sem'][start:end]

    diff =  pool0 - pool1
    sem_sum = sem0+sem1
    sig_diff = np.abs(diff) > sem_sum

    t = np.linspace(t1, t2, len(diff))
    axes[1].plot(t, diff, label='{} minus {}'.format(pools[0], pools[1]), color='black')
    axes[1].fill_between(t, -sem_sum, sem_sum, alpha=0.5, color='gray')
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].legend(prop={'size': 15})
    axes[1].set_xlabel('time (s)', fontsize=18)
    axes[1].set_ylabel('euclidean distance', fontsize=18)
    axes[1].tick_params(axis='both', which='major', labelsize=15)
    fig.suptitle('site {}, {}, probe {}'.format(site, modelname, probe))

    fig.suptitle('site {}, probe {}'.format(site, probe), fontsize=20)
    fig.set_size_inches(20,10)

    # fig.savefig('/home/mateo/Pictures/WIP2/181212_eg_{}_{}_{}_vs_{}_closeup.png'.format(site, probe, pools[0], pools[1]), dpi=300)

########################################################################################################################
# prints close up of all the sites, side by side

start = 290
end = 600
t1 = (start / 100) - 3
t2 = (end / 100) - 3


traces = dict()
errors = dict()
for site in ['BRT055b', 'BRT054b', 'BRT056b']:
    modelname = 'resp'
    sig = pop_sigs[site][modelname]
    ctxs = prbs = [0, 1, 2, 3, 4]
    pooled_dict = cdisp.signal_single_trial_dispersion_pooled(signal=sig, probe_names=ctxs, context_names=prbs,
                                                              shuffle_neurons=False,
                                                              signal_name='{}_{}'.format(site, modelname),
                                                              recache=False)

    mean0 = pooled_dict['P_all']['diff_ctx']['mean']
    sem0 = pooled_dict['P_all']['diff_ctx']['sem']

    mean1 = pooled_dict['P_all']['same_ctx']['mean']
    sem1 = pooled_dict['P_all']['same_ctx']['sem']

    traces[site] = mean0 - mean1
    errors[site] = sem0 + sem1


fig, ax = plt.subplots()
y_tics = list()
y_lab = list()

for ii,(key, trace) in enumerate(traces.items()):

    y_off = ii * 0.5

    line = traces[key][start:end]
    shade = errors[key][start:end]
    # defiens y_label positions and values

    zero = y_off
    max_y = y_off + np.nanmax(line)
    y_tics.append(zero)
    y_tics.append(max_y)
    y_lab.append(0)
    y_lab.append('{:.2f}'.format(np.nanmax(line)))

    t = np.linspace(t1, t2, len(line))
    ax.plot(t, line + y_off, label='{}'.format(key), color='C{}'.format(ii+5))
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

fig.savefig('/home/mateo/Pictures/WIP2/181213_all_sites_diff_min_same_ctx_all_prb_closeup.png', dpi=300)


########################################################################################################################
# new significance calculation, using shuffle test across contexts
ctxs = [0, 1, 2, 3, 4]
prbs = [1, 2, 3, 4]
modelname = 'resp'

site_shuffled_disp = coll.defaultdict(dict)
for site in ['BRT055b', 'BRT054b', 'BRT056b']:
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
end = 600
t1 = (start / 100) - 3
t2 = (end / 100) - 3

fig, ax = plt.subplots()

for ii,(site, vals) in enumerate(site_shuffled_disp.items()):

    line = vals['real'][start:end]
    shade = vals['distribution'][:, start:end]
    shade = np.mean(shade,axis=0)+np.std(shade,axis=0)*2


    y_off = ii * 0.5

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

fig.savefig('/home/mateo/Pictures/WIP2/181213_all_sites_diff_min_same_ctx_all_prb_closeup.png', dpi=300)



