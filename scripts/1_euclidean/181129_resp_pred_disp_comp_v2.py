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

################################################
# iterates over each site id, and calculates the dispersion for predictions of each different model
site_disps = dict()
site_pvals = dict()
site_cnames = dict()
for disp_val in ['metric', 'pvalue']:
    for site, recs in pop_recs.items():
        # for multi core running
        # site_IDs = list(pop_recs.keys())
        # site = site_IDs[4]
        # recs = pop_recs[site]

        formated = {key: cep.set_recording_subepochs(rec) for key, rec in recs.items()}

        # calculatest the dispersion for each prediction and one response
        dispersions = dict()
        c_names = dict()
        for modelname, rec in formated.items():
            dispersions[modelname], c_names[modelname] = cdisp.signal_all_context_sigdif(rec['pred'], channels='all',
                                                                  signal_name='{}_{}'.format(modelname, site),
                                                                  probes=(1, 2, 3, 4), dimensions='population', sign_fs=100,
                                                                  window=1, rolling=True, type='Euclidean', recache=False,
                                                                  value=disp_val)

        real_neu_resp = formated['wc.2x2.c-fir.2x15-lvl.1-dexp.1']['resp']
        dispersions['resp'], c_names['resp'] = cdisp.signal_all_context_sigdif(real_neu_resp, channels='all',
                                                              signal_name='{}_{}'.format('resp', site),
                                                              probes=(1, 2, 3, 4), dimensions='population', sign_fs=100,
                                                              window=1, rolling=True, type='Euclidean', recache=False,
                                                              value=disp_val)

        if disp_val == 'metric':
            site_disps[site] = dispersions
        elif disp_val == 'pvalue':
            site_pvals[site] = dispersions
        site_cnames[site] = c_names

del(site_disps['BRT057b'])
del(site_pvals['BRT057b'])
del(site_cnames['BRT057b'])
################################################
# plots the dispersion over time for each site, including the predicted responses
for site_ID, models in site_disps.items():
    # fits an exponential decay. Todo make it work
    # fitted = {key: cdisp.disp_exp_decay(val, start=300, prior=1, axis=None)[0] for key, val in collapsed.items()}

    fig, ax = plt.subplots()
    x_ax = np.linspace(-3,3, site_disps[site_ID]['resp'].shape[1])
    for ii, (model, matrix) in enumerate(models.items()):
        label = shortnames[model]
        color = color_dict[model]
        mean = np.mean(matrix,axis=0)
        ax.plot(x_ax, mean, label=label, color=color)
        ax.plot(x_ax, matrix.T, color=color, alpha=0.2)
        # ax.plot(fitted[key], color=color)
    ax.axvline(0, color='black')
    ax.legend()
    ax.set_xlabel('Time S')
    ax.set_ylabel('euclideand distance\nspikes/s')
    fig.suptitle('dispersion over time\nsite: {}'.format(site_ID))
    fig.set_size_inches(8,5)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_dist_over_time_{}_resp_pred.png'.format(site_ID), dpi=100)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_dist_over_time_{}_resp_pred.svg'.format(site_ID))
plt.close('all')



################################################
# plots the dispersion over time for each site response, pooled together
fig, ax = plt.subplots()
for cc, (site_ID, models) in enumerate(site_disps.items()):
    label = site_ID
    color = 'C' + str(cc)
    matrix = models['resp']
    mean = np.mean(matrix,axis=0)
    x_ax = np.linspace(-3,3, len(mean))
    ax.plot(x_ax, mean, label=label, color=color)
    ax.plot(x_ax, matrix.T, color=color, alpha=0.2)

ax.axvline(0, color='black')
ax.legend()
ax.set_xlabel('Time S')
ax.set_ylabel('euclideand distance\nspikes/s')
fig.suptitle('dispersion over time of response for all sites')
fig.set_size_inches(8,5)
fig.savefig('/home/mateo/Pictures/DAC1/181205_dist_over_time_all_mod_resp.png', dpi=100)
fig.savefig('/home/mateo/Pictures/DAC1/181205_dist_over_time_all_mod_resp.svg')
plt.close('all')

################################################
# plots the dispersion over time for each site predictions, excluding the response and the stim_num
for site_ID, models in site_disps.items():
    # fits an exponential decay. Todo make it work
    # fitted = {key: cdisp.disp_exp_decay(val, start=300, prior=1, axis=None)[0] for key, val in collapsed.items()}

    fig, ax = plt.subplots()
    for ii, (model, matrix) in enumerate(models.items()):
        if model == 'resp':
            continue
        label = shortnames[model]
        color = color_dict[model]
        middle = int(np.floor(matrix.shape[1]/2))
        matrix = matrix[:, middle:]
        mean = np.mean(matrix, axis=0)
        x_ax = np.linspace(0, 3, len(mean))
        ax.plot(x_ax, mean, label=label, color=color)
        ax.plot(x_ax, matrix.T, color=color, alpha=0.2)
        # ax.plot(fitted[key], color=color)
    ax.axvline(0, color='black')
    ax.legend()
    ax.set_xlabel('Time S')
    ax.set_ylabel('euclideand distance\nspikes/s')
    fig.suptitle('model predictions dispersion over time\nsite: {}'.format(site_ID))
    fig.set_size_inches(8,5)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_dist_over_time_{}_pred.png'.format(site_ID), dpi=100)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_dist_over_time_{}_pred.svg'.format(site_ID))
plt.close('all')


################################################
# plots the max pvalue of the dispersions over time
for site_ID, models in site_pvals.items():
    # fits an exponential decay. Todo make it work
    # fitted = {key: cdisp.disp_exp_decay(val, start=300, prior=1, axis=None)[0] for key, val in collapsed.items()}

    fig, ax = plt.subplots()
    for ii, (model, matrix) in enumerate(models.items()):
        label = shortnames[model]
        color = color_dict[model]
        middle = int(np.floor(matrix.shape[1]/2))
        mean = np.max(matrix, axis=0)
        x_ax = np.linspace(-3, 3, len(mean))
        ax.plot(x_ax, mean, label=label, color=color)
        # ax.plot(x_ax, matrix.T, color=color, alpha=0.2)
    ax.axvline(0, color='black')
    ax.axhline(0.05, color='black')
    ax.legend()
    fig.suptitle('max pvalue of difference \nsite: {}'.format(site_ID))
    fig.set_size_inches(8,5)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_pvalue_max_{}.png'.format(site_ID), dpi=100)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_pvalue_max_{}.svg'.format(site_ID))
plt.close('all')


################################################
# plots the Min pvalue of the dispersions over time
for site_ID, models in site_pvals.items():
    # fits an exponential decay. Todo make it work
    # fitted = {key: cdisp.disp_exp_decay(val, start=300, prior=1, axis=None)[0] for key, val in collapsed.items()}

    fig, ax = plt.subplots()
    for ii, (model, matrix) in enumerate(models.items()):
        label = shortnames[model]
        color = color_dict[model]
        middle = int(np.floor(matrix.shape[1]/2))
        mean = np.min(matrix, axis=0)
        x_ax = np.linspace(-3, 3, len(mean))
        ax.plot(x_ax, mean, label=label, color=color)
        # ax.plot(x_ax, matrix.T, color=color, alpha=0.2)
    ax.axvline(0, color='black')
    ax.axhline(0.05, color='black')
    ax.legend()
    fig.suptitle('min pvalue of difference \nsite: {}'.format(site_ID))
    fig.set_size_inches(8,5)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_pvalue_min_{}.png'.format(site_ID), dpi=100)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_pvalue_min_{}.svg'.format(site_ID))
plt.close('all')


################################################
# plots examples of calculation Process
best_site = 'BRT056b'
best_cell = 'BRT056b-58-1'
best_voc = None
best_cont = None

# full population, stim_num independent PSTH vocalization
sig = pop_sigs[best_site]['resp']
fig, axes = cplot.hybrid(sig,scatter_kws={'alpha': 0.1})
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
fig.suptitle('stim_num independent vocalizations PSTH\nsite {}'.format(best_site))
fig.savefig('/home/mateo/Pictures/DAC1/181205_voc_psth_all_cells_{}.png'.format(best_site), dpi=100)
fig.savefig('/home/mateo/Pictures/DAC1/181205_voc_psth_all_cells_{}.svg'.format(best_site))


# single cell, all vocalizations, all contexts
cpp_eps = nep.epoch_names_matching(sig.epochs, r'\AC\d_P\d$')

for ii in range(1,5):
    fig, axes = cplot.hybrid(sig, epoch_names=r'\AC\d_P{}'.format(ii), channels=[best_cell],
                 start=3, end=6,scatter_kws={'alpha': 0.2})
    fig.suptitle('stim_num dependent response to prb {}'.format(ii))
    fig.set_size_inches(9,6)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_CPP_probe{}_PSTH_cell_{}.png'.format(ii, best_cell), dpi=100)
    # fig.savefig('/home/mateo/Pictures/DAC1/181205_CPP_probe{}_PSTH_cell_{}.svg'.format(ii, best_cell))

# plots dispersion over time for a single prb with all the contexts
fig, ax = plt.subplots()
labels = site_cnames[best_site]['resp']
matrix = site_disps[best_site]['resp']

for ii in range(matrix.shape[0]):
    label = labels[ii]
    color = voc_color['voc_{}'.format(label[-1])]
    x_ax = np.linspace(-3,3, matrix.shape[1])
    ax.plot(x_ax, matrix[ii,:] + ii , color=color, label=label)
ax.axvline(0, color='black')
ax.legend()
ax.set_xlabel('Time S')
ax.set_ylabel('euclideand distance\nspike rate')
fig.suptitle('dispersion over time of response for each prb\nsite {}'.format(best_site))
fig.set_size_inches(8,5)
fig.savefig('/home/mateo/Pictures/DAC1/181205_disp_by_probe_{}.png'.format(best_site), dpi=100)
# fig.savefig('/home/mateo/Pictures/DAC1/181205_disp_by_probe_{}.svg'.format(best_site))


################################################
# cleaner version of example for wip talk

sig = pop_sigs[best_site]['resp']
fig, axes = cplot.hybrid(sig, epoch_names=r'\AC\d_P3', start=3, end=6, scatter_kws={'alpha': 0.1})
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.tight_layout()
for ax in axes: ax.axis('off')
fig.suptitle('')
# fig.suptitle('stim_num independent vocalizations PSTH\nsite {}'.format(best_site))
fig.savefig('/home/mateo/Pictures/WIP2/181207_eg_cpp_all_cells_{}.png'.format(best_site), dpi=100)
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
fig.savefig('/home/mateo/Pictures/WIP2/181205_CPP_probe{}_PSTH_cell_{}.png'.format(goodprobe, goodcell), dpi=200)
# fig.savefig('/home/mateo/Pictures/DAC1/181205_CPP_probe{}_PSTH_cell_{}.svg'.format(ii, best_cell))



# single cell, one prb, all contexts, close up in time
goodprobe = 3
goodcell = 'BRT056b-58-1'
fig, axes = cplot.hybrid(sig, epoch_names=r'\AC\d_P{}'.format(goodprobe), channels=[goodcell],
             start=None, end=None,scatter_kws={'alpha': 0.4}, time_strech=[3, 3.5])
for ax in axes: ax.axis('off'); ax.set_title('')
fig.suptitle('')
fig.set_size_inches(13,9)
plt.tight_layout()
fig.savefig('/home/mateo/Pictures/WIP2/181205_CPP_probe{}_PSTH_cell_{}_closeup.png'.format(goodprobe, goodcell), dpi=200)
# fig.savefig('/home/mateo/Pictures/DAC1/181205_CPP_probe{}_PSTH_cell_{}.svg'.format(ii, best_cell))


# single cell, one prb, all contexts, close up in time. Similar as previous plot but with only two overlayed examples







