import collections as col
import itertools as itt

import cpp_cache as ccache
import cpp_dispersion as cdisp
import cpp_epochs as cep
import cpp_reconstitute_rec as crec
import cpp_PCA as cpca
import cpp_plots as cplt
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.signal as ssig
import scipy.ndimage.filters as sf


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


sites = crec.get_site_ids(310)

pop_recs = col.defaultdict(dict)
for (site_name, cells), modelname in itt.product(sites.items(), all_models):
    print('#####\nreconstituting site {} with model {}\n '.format(site_name, modelname))
    recons_args = {'batch':310, 'cellid_list':cells, 'modelname': modelname}
    recons_cache = ccache.make_cache(crec.reconsitute_rec, func_args=recons_args, classobj_name='reconstitution',
                                     recache=False, cache_folder='/home/mateo/mycache/reconstitute_recs',
                                     use_hash=True)
    reconstituted_recording = ccache.get_cache(recons_cache)
    pop_recs[site_name][modelname] = reconstituted_recording
    
# removese the flat response of BRT057b todo solve the bug
del(pop_recs['BRT057b'])

# reorders in dictionary of signals, including only the response and the prediction of each mode
# reformats the epochs

pop_sigs = col.defaultdict(dict)
for site_key, model_recs in pop_recs.items():
    for modelname, rec in model_recs.items():
        rec = cep.set_recording_subepochs(rec, set_pairs=False)
        pop_sigs[site_key][modelname] = rec['pred'].rasterize()
    pop_sigs[site_key]['resp'] = rec['resp'].rasterize()
    

# calculates the PCAs for each site, for each model prediction and response
PCA_stats = col.defaultdict(dict)
site_PCA = col.defaultdict(dict)
for site_key, site_models in pop_sigs.items():
    for sig_key, signal in site_models.items():
        PCA_sig, stats = cpca.signal_PCA(signal)
        site_PCA[site_key][sig_key] = PCA_sig
        PCA_stats[site_key][sig_key] = stats

# plots the variance explained by PCs for each site
for site_key, site_vals in PCA_stats.items():
    fig, ax = plt.subplots()
    for sig_key, pca_stats in site_vals.items():
        toplot = np.cumsum(pca_stats.explained_variance_ratio_)
        label = shortnames[sig_key]
        ax.plot(toplot, '.-', color=color_dict[sig_key], label=label)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('cumulative variance explained')
        ax.legend()
        ax.set_title('site {}'.format(site_key))
    fig.set_size_inches(4,4)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_PCA_explained_var_{}.png'.format(site_key), dpi=100)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_PCA_explained_var_{}.svg'.format(site_key))
    plt.close('all')

# plot example 3D trajectory for BRT056b PCA
site = 'BRT056b'
sig = site_PCA[site]['resp']

# plots all the vocalizations, independent of context
fig = plt.figure()
ax = fig.gca(projection='3d')
for context in range(1, 5):
    if context == 0:
        epoch = 'PostStimSilence'
    else:
        epoch = 'voc_{}'.format(context)

    matrix = sig.extract_epoch(epoch)
    matrix = ssig.decimate(matrix, 1, axis=2)
    matrix_mean = np.mean(matrix, axis=0)

    # smooths the matrixes  with a gaussian filter along the time dimention
    matrix = sf.gaussian_filter(matrix, [0, 0, 3])
    matrix_mean = sf.gaussian_filter(matrix_mean, [0, 3])

    N = matrix.shape[2]

    # starts at blue, ends at yellow
    cm = plt.cm.get_cmap(voc_cmpat['voc_{}'.format(context)]+'_r')
    cmap = cm(np.linspace(0, 1, N))

    # plot the mean
    x = matrix_mean[0, :]
    y = matrix_mean[1, :]
    z = matrix_mean[2, :]

    for ii in range(N - 1):
        p = ax.plot(x[ii:ii + 2], y[ii:ii + 2], z[ii:ii + 2], color=cmap[ii, :], alpha=1)

ax.set_xlabel('component 1')
ax.set_ylabel('component 2')
ax.set_zlabel('component 3')
fig.suptitle('{} all vocalizations, context independent'.format(site))
fig.set_size_inches(10,10)
fig.savefig('/home/mateo/Pictures/DAC1/181205_3D_{}_all_vocs.png'.format(site), dpi=100)
fig.savefig('/home/mateo/Pictures/DAC1/181205_3D_{}_all_vocs.svg'.format(site))
plt.close('all')

# probes after different context
for probe in range(5):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for context in range(5):
        if probe ==0:
            if context == 0:
                continue
        matrix = sig.extract_epoch('C{}_P{}'.format(context, probe))
        matrix = ssig.decimate(matrix, 1, axis=2)
        matrix_mean = np.mean(matrix, axis=0)

        # smooths the matrixes  with a gaussian filter along the time dimention
        matrix = sf.gaussian_filter(matrix, [0, 0, 3])
        matrix_mean = sf.gaussian_filter(matrix_mean, [0, 3])

        N = matrix.shape[2]

        # starts at blue, ends at yellow
        cm = plt.cm.get_cmap(voc_cmpat['voc_{}'.format(context)]+'_r')
        cmap = cm(np.linspace(0, 1, N))

        # plot the mean
        x = matrix_mean[0, :]
        y = matrix_mean[1, :]
        z = matrix_mean[2, :]

        for ii in range(N - 1):
            p = ax.plot(x[ii:ii + 2], y[ii:ii + 2], z[ii:ii + 2], color=cmap[ii, :], alpha=1)

    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.set_zlabel('component 3')
    fig.suptitle('{} vocalization {}, context dependent'.format(site, probe))
    fig.set_size_inches(10,10)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_3D_{}_probe{}.png'.format(site,probe), dpi=100)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_3D_{}_probe{}.svg'.format(site,probe))
    plt.close('all')

# repreats the context figures but onlye the initial 500 ms, i.e first 50 time bins

for probe in range(5):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for context in range(5):
        if probe ==0:
            if context == 0:
                continue
        matrix = sig.extract_epoch('C{}_P{}'.format(context, probe))
        matrix = matrix[:, :, :50]
        matrix = ssig.decimate(matrix, 1, axis=2)
        matrix_mean = np.mean(matrix, axis=0)

        # smooths the matrixes  with a gaussian filter along the time dimention
        matrix = sf.gaussian_filter(matrix, [0, 0, 3])
        matrix_mean = sf.gaussian_filter(matrix_mean, [0, 3])

        N = matrix.shape[2]

        # starts at blue, ends at yellow
        cm = plt.cm.get_cmap(voc_cmpat['voc_{}'.format(context)]+'_r')
        cmap = cm(np.linspace(0, 1, N))

        # plot the mean
        x = matrix_mean[0, :]
        y = matrix_mean[1, :]
        z = matrix_mean[2, :]

        for ii in range(N - 1):
            p = ax.plot(x[ii:ii + 2], y[ii:ii + 2], z[ii:ii + 2], color=cmap[ii, :], alpha=1)

    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.set_zlabel('component 3')
    fig.suptitle('{} vocalization {}, context dependent'.format(site, probe))
    fig.set_size_inches(10,10)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_3D_start_{}_probe{}.png'.format(site,probe), dpi=100)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_3D_start_{}_probe{}.svg'.format(site,probe))
    plt.close('all')


    

