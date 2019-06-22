import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import nems.db as nd
import nems.recording as recording
import nems_lbhb.baphy as nb

import cpp_epochs as cpe
# import cpp_dispersion as cdisp
import cpp_cache as cch
import cpp_plots as cplot
import cpp_PCA as cpca

import cpn_triplets as tp
import cpp_dispersion as cpd
import cpn_dispersion as cdisp


# harcoded selected cells as taken from 190529_select_single_cells_with_high_variation.py
# using zscore threshold = 0.18, and zstd (zscore std across contexts) = 0

selected_cells = ['ley070a-01-1', 'ley070a-01-2', 'ley070a-07-1', 'ley070a-37-1', 'ley072b-25-1', 'ley072b-27-1',
                  'ley072b-28-1', 'ley072b-30-1', 'ley072b-30-2', 'ley072b-34-1', 'AMT028b-07-2', 'AMT028b-13-1',
                  'AMT028b-30-1', 'AMT028b-34-1', 'AMT028b-40-1', 'AMT028b-43-1', 'AMT028b-48-1', 'AMT029a-06-1',
                  'AMT029a-27-1', 'AMT029a-34-1', 'AMT029a-38-1', 'AMT029a-43-1', 'AMT029a-48-1', 'AMT029a-48-3',
                  'AMT029a-51-1', 'AMT029a-54-1', 'AMT029a-55-1', 'AMT029a-57-1', 'AMT029a-60-1', 'AMT030a-24-1',
                  'AMT030a-27-2', 'AMT030a-28-1', 'AMT030a-30-1', 'AMT030a-30-2', 'AMT032a-32-1', 'AMT032a-33-2',
                  'AMT032a-39-1', 'AMT032a-49-1', 'AMT032a-51-1', 'AMT032a-51-2', 'AMT032a-52-1']


# from the selectee cells, get the sites. loads and analyzese only for that subset of cells
selected_sites = set([cellid[0:7] for cellid in selected_cells])

# for site in selected_sites:
site = 'AMT029a'
modelname = 'resp'
options = {'batch': 316,
           'siteid': site,
           'stimfmt': 'envelope',
           'rasterfs': 100,
           'recache': False,
           'runclass': 'CPN',
           'stim': False}  # ToDo chace stims, spectrograms???


load_URI = nb.baphy_load_recording_uri(**options)
loaded_rec = recording.load_recording(load_URI)

rec = cpe.set_recording_subepochs(loaded_rec, set_pairs=True)
sig = rec['resp'].rasterize()
eps = sig.epochs


probe_names = [6, 7, 9, 10]
ctx_transitions = ['silence', 'continuous']

goodcells = [gcell for gcell in selected_cells if gcell.split('-')[0]==site]

real, shuffled = cdisp.signal_single_trial_dispersion_pooled_shuffled(sig, probe_names, ctx_transitions, channels=goodcells, shuffle_num=10,
                                                           trial_combinations=True)

def plot_dist_with_CI(real, shuffled, start, end, fs, suptitle):

    t1 = (start / fs) - 1
    t2 = (end / fs) - 1
    fig, ax = plt.subplots()
    line = real[start:end]
    shade = shuffled[:, start:end]
    shade = np.mean(shade, axis=0) + np.std(shade, axis=0) * 2
    t = np.linspace(t1, t2, len(line))
    ax.plot(t, line, label='{}'.format(site), color='C0')
    ax.fill_between(t, -shade, shade, alpha=0.5, color='C0')
    ax.axvline(0, color='black', linestyle='--')
    # ax.legend(prop={'size': 15})

    ax.set_xlabel('time (s)', fontsize=18)
    ax.set_ylabel('euclidean distance', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=15)
    fig.suptitle(suptitle, fontsize=20)
    fig.set_size_inches(20, 10)

    return fig, ax

plot_dist_with_CI(real, shuffled, start=0, end=200, fs=100, suptitle=f'AMT029, good cells, {ctx_transitions}')

########################################################################################################################
# population analysis similar to that done for WIP talk 2. in short comparing difference between paris of CPPs, in a tria
# by trial basis, and normalizing by the difference between trials from the same CPP
# in addition this can be done specifying a subset of transitions type between context and probe. the recomendation is to
# use pairs of transition types, for example compare the difference between transitions from silence and from a continuos
# sound


interesting_transition_pairs = [['silence', 'continuous'], ['silence', 'sharp'], ['continuous', 'sharp'],
                                ['continuous', 'similar']]


for int_tran in interesting_transition_pairs:

    for site in selected_sites:

        goodcells = [gcell for gcell in selected_cells if gcell.split('-')[0] == site]

        for what_cells in [goodcells, 'all']:

            prbs = [6, 7, 9, 10]
            modelname = 'resp'

            options = {'batch': 316,
                       'siteid': site,
                       'stimfmt': 'envelope',
                       'rasterfs': 100,
                       'recache': False,
                       'runclass': 'CPN',
                       'stim': False}  # ToDo chace stims, spectrograms???

            load_URI = nb.baphy_load_recording_uri(**options)
            loaded_rec = recording.load_recording(load_URI)

            rec = cpe.set_recording_subepochs(loaded_rec, set_pairs=True)
            sig = rec['resp'].rasterize()


            start = 0
            end = 200

            ##
            signal_name = '190505_{}_{}'.format(site, modelname)

            func_args = {'signal': sig, 'probe_names': prbs, 'context_transitions': int_tran, 'channels': what_cells,
                         'shuffle_num': 1000, 'trial_combinations': True}

            shuffled_dispersion_time = cch.make_cache(function=cdisp.signal_single_trial_dispersion_pooled_shuffled,
                                                      func_args=func_args,
                                                      classobj_name=signal_name, recache=False,
                                                      cache_folder='/home/mateo/mycache/shuffled_euclideans')
            real, shuffled = cch.get_cache(shuffled_dispersion_time)


########################################################################################################################
# plots only a subset
site = 'AMT029a'


for int_tran in interesting_transition_pairs:

    options = {'batch': 316,
               'siteid': site,
               'stimfmt': 'envelope',
               'rasterfs': 100,
               'recache': False,
               'runclass': 'CPN',
               'stim': False}  # ToDo chace stims, spectrograms???

    load_URI = nb.baphy_load_recording_uri(**options)
    loaded_rec = recording.load_recording(load_URI)
    rec = cpe.set_recording_subepochs(loaded_rec, set_pairs=True)
    sig = rec['resp'].rasterize()

    signal_name = '190505_{}_{}'.format(site, 'resp')

    func_args = {'signal': sig, 'probe_names': [6, 7, 9, 10], 'context_transitions': int_tran, 'channels': 'all',
                 'shuffle_num': 1000, 'trial_combinations': True}

    shuffled_dispersion_time = cch.make_cache(function=cdisp.signal_single_trial_dispersion_pooled_shuffled,
                                              func_args=func_args,
                                              classobj_name=signal_name, recache=False,
                                              cache_folder='/home/mateo/mycache/shuffled_euclideans')
    real, shuffled = cch.get_cache(shuffled_dispersion_time)


    suptitle = f'{site}, all, {int_tran}'
    plot_dist_with_CI(real, shuffled, start=0, end=200, fs=100, suptitle=suptitle)

########################################################################################################################


