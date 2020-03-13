import itertools as itt

import matplotlib.pyplot as plt
import numpy as np
from progressbar import ProgressBar

import cpn_LDA as cLDA
import cpn_dPCA as cdPCA
import cpn_dprime as cDP
import nems.recording as recording
import nems_lbhb.baphy as nb
from reliability import signal_reliability
from tools import shuffle_along_axis as shuffle
from cpp_cache import make_cache, get_cache
from fancy_plots import _cint
from nems import db as nd
from nti_arrays import raster_from_sig
from nti_epochs import set_recording_subepochs, NTI_epoch_name

'''
first attemtp at using the dprime contextual effect analysis with Sams NTI data. after some preprosecing to get the 
adequate rasters for LDA (and potentially dPCA) selects a given probe, finds all its potential contexts and performs an 
n-way LDA (trying to discriminate all contexts simultaneously). 
Uses then the projection to the main LDA axis to determine the dprimer between pairs of context-probes.

the results are not super promissing, part of the big differences between the CPP/CPN and the NTI data is the 
diversity vs repetitions (CPP/CPN: low diversity, high reps ; NTI: high diversity, low reps)

aditionally, the shorters segments i.e. 16ms are too short to analyse
'''

batch = 319  # NTI batch, Sam paradigm
# check sites in batch
batch_cells = nd.get_batch_cells(batch)
cell_ids = batch_cells.cellid.unique().tolist()
site_ids = set([cellid.split('-')[0] for cellid in cell_ids])

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'significance': False,
        'montecarlo': 1000,
        'zscore': False}

code_to_name = {'t': 'Probe', 'ct': 'Context'}


########################################################################################################################
def nway_analysis(full_raster, meta):
    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, _, _ = cdPCA.format_raster(full_raster)
    trialR = trialR.squeeze()  # squeezes out probe
    R, C, S, T = trialR.shape

    # calculates full LDA. i.e. considering all 4 categories
    LDA_projection, LDA_transformation = cLDA.fit_transform_over_time(trialR, 1)
    dprime = cDP.pairwise_dprimes(LDA_projection.squeeze())

    # calculates floor (ctx shuffle) and ceiling (simulated data)
    sim_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))
    shuf_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))

    ctx_shuffle = trialR.copy()

    pbar = ProgressBar()
    for rr in pbar(range(meta['montecarlo'])):
        # ceiling: simulates data, calculates dprimes
        sim_trial = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0),
                                     size=[R, C, S, T])
        sim_projection = cLDA.transform_over_time(cLDA._reorder_dims(sim_trial), LDA_transformation)
        sim_dprime[rr, ...] = cDP.pairwise_dprimes(cLDA._recover_dims(sim_projection).squeeze())

        ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)
        shuf_projection, _ = cLDA.fit_transform_over_time(ctx_shuffle)
        shuf_dprime[rr, ...] = cDP.pairwise_dprimes(shuf_projection.squeeze())

    return dprime, shuf_dprime, sim_dprime


def dPrime_from_NIT_site(site, duration, source, position, meta):
    options = {'batch': batch,
               'siteid': site,
               'stimfmt': 'envelope',
               'rasterfs': 100,
               'recache': False,
               'runclass': 'NTI',
               'stim': False}
    load_URI = nb.baphy_load_recording_uri(**options)
    rec = recording.load_recording(load_URI)

    rec = set_recording_subepochs(rec)
    sig = rec['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    probe_regex = NTI_epoch_name(duration, source, position)
    cp_regex = fr'\AC(({NTI_epoch_name()})|(PreStimSilence))_P{probe_regex}\Z'

    full_rast, transitions, contexts = raster_from_sig(sig, cp_regex, goodcells)

    if len(contexts) < 2:
        real = shuffled = simulated = None
        print(f'only one context for {probe_regex}, skiping analysis')
    else:
        real, shuffled, simulated = nway_analysis(full_rast, meta)

    return real, shuffled, simulated, transitions, contexts


########################################################################################################################


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',  # blue, orange, green, brow,
                  '#984ea3', '#999999', '#e41a1c', '#dede00']  # purple, gray, scarlet, lime

trans_color_map = {'silence': '#377eb8',  # blue
                   'continuous': '#ff7f00',  # orange
                   'similar': '#4daf4a',  # green
                   'sharp': '#a65628'}  # brown

ci_color = {'shuffled': 'orange',
            'simulated': 'purple'}

# transferable plotting parameters
plt.rcParams['svg.fonttype'] = 'none'
sup_title_size = 15
sub_title_size = 12
ax_lab_size = 15
ax_val_size = 11

########################################################################################################################

site = 'AMT028b'  # example site
options = {'batch': batch,
           'siteid': site,
           'stimfmt': 'envelope',
           'rasterfs': 100,
           'recache': False,
           'runclass': 'NTI',
           'stim': False}
load_URI = nb.baphy_load_recording_uri(**options)
rec = recording.load_recording(load_URI)

rec = set_recording_subepochs(rec)
sig = rec['resp']

# calculates response realiability and select only good cells to improve analysis
r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
goodcells = goodcells.tolist()

######### generates an array ordering a probe and its contexts. Defines the kind of transitions for such cp combinations
# [C, P, R, U, T] final array dimesions


duration = 250
source = 19
position = 1

site = 'AMT028b'  # example site

for source in range(20):

    probe = NTI_epoch_name(duration, source, position)

    fourway_name = f'191014_{site}_P{probe}_fourway_analysis'
    analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
    analysis_name = 'NTI_LDA_dprime'

    nway = make_cache(function=dPrime_from_NIT_site,
                      func_args={'site': site, 'duration': duration, 'source': source,
                                 'position': position, 'meta': meta},
                      classobj_name=fourway_name,
                      cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}')

    real, shuffled, simulated, transitions, contexts = get_cache(nway)  # get_cache(twoway)

    if real is None:
        continue

    # pairwise dprime
    trans_pairs = list(itt.combinations(range(len(transitions)), 2))
    trans_count = 0

    nrow = 1
    ncol = len(trans_pairs)
    fig, axes = plt.subplots(nrow, ncol, sharey=True, sharex=True, squeeze=False)
    for row, col in itt.product(range(nrow), range(ncol)):
        trans_pair = trans_pairs[trans_count]

        ax = axes[row, col]
        T = real.shape[1]
        time = np.linspace(0, T / meta['raster_fs'], T, endpoint=False)

        ci = 0.9
        ax.plot(time, np.mean(shuffled[:, trans_count, :], axis=0), color=ci_color['shuffled'], alpha=1)
        _cint(time, shuffled[:, trans_count, :], ci, ax=ax,
              fillkwargs={'alpha': 0.5, 'color': ci_color['shuffled'], 'label': 'context discrimination'})
        # ceiling
        ax.plot(time, np.mean(simulated[:, trans_count, :], axis=0), color=ci_color['simulated'], alpha=1)
        _cint(time, simulated[:, trans_count, :], ci, ax=ax,
              fillkwargs={'alpha': 0.5, 'color': ci_color['simulated'], 'label': 'population effect'})
        # real dprime
        ax.plot(time, real[trans_count, :], color='black', linestyle='-', label='real value')

        # Formatting
        ax.tick_params(labelsize=ax_val_size)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.tick_params(labelsize=ax_val_size)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # distance labels only on leftmost plots
        if col == 0:
            ax.set_ylabel(f"d'", fontsize=ax_lab_size)
            # trial_ax.set_ylabel(f'euclidean distance(Hz)', fontsize=20)
            pass
        else:
            ax.axes.get_yaxis().set_visible(False)

        ax.set_title(f'{transitions[trans_pair[0]]} vs {transitions[trans_pair[1]]}', fontsize=sub_title_size)

        # time only in bottom plot
        if row != nrow - 1:
            ax.axes.get_xaxis().set_visible(False)
        else:
            ax.set_xlabel('time (s)', fontsize=ax_lab_size)

        # adds legend in the last ax
        if trans_count + 1 == nrow * ncol:
            ax.legend()

        trans_count += 1

    a_name = 'n-way'

    suptitle = f"{site} probe {probe} {a_name} LDA dprime {meta['raster_fs']}Hz zscore {meta['zscore']}"
    fig.suptitle(suptitle, fontsize=sup_title_size)

    analysis = f"LDA_{a_name}_{meta['raster_fs']}Hz_zscore-{meta['zscore']}"
