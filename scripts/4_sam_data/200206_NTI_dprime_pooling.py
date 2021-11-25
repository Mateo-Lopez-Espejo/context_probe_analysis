import itertools as itt
import collections as coll
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
from progressbar import ProgressBar

from src.data import LDA as cLDA, dPCA as cdPCA
from src.metrics import dprime as cDP
import nems.recording as recording
import nems_lbhb.baphy as nb
from src.metrics.reliability import signal_reliability
from src.utils.tools import shuffle_along_axis as shuffle
from src.data.cache import make_cache, get_cache
from src.visualization.fancy_plots import _cint
from nems import db as nd
from src.data.nti_arrays import raster_from_sig
from src.data.nti_epochs import set_recording_subepochs, NTI_epoch_name, _nom2real_dur

'''
after a first pass with the dprime analysis on Sam's NTI data, it was clear that this data compromises number of contexts
and repetitions for a broader selection of probes. This lead to a rather noisy result, given the limited number of repetitions
however this could potentially be circumvented by pooling acroos the numerous probes. this script attempts to do so.
1. for each pronbe and its contexts, transition categories are defined (silence, sharp, continuous). 
2. pairwise dprime and montecarlo shuffled and simulated dprimes are calculated. defining the pair of contexts compared
e.g continuous_silence
3. for each transition pair category, the dprimes, shuffled and simulated values are pooled across all different probes 
4. the pooled dprime is definesd as the mean of dprimes across probes
the shuffled and simulated are pooled across probes, and the probe dimention is collapsed into the repetitions

afterword:
depending on the duration of the segments used, the number or available repetitions change, thus with longer segments and
therefore less repetitions (as lowe as 3) the LDA and dprime analysis starts to break. For the LDA this happens as 
the algorithm can more easily find a projection that minimizes the variance within category to values close to 0. 
this in turn propagates to the dprime, leading to artificially super high values. Not only this, given the parametric nature
of the dprime analysis, 3 repetitions are by no means adequate.

Some sites might work much better given that they have a greater number of repetitions
'''

batch = 319  # NTI batch, Sam paradigm
load_fs = 100 # sampling freq of loaded signal
# check sites in batch
batch_cells = nd.get_batch_cells(batch)
cell_ids = batch_cells.cellid.unique().tolist()
site_ids = set([cellid.split('-')[0] for cellid in cell_ids])

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 100,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'significance': False,
        'montecarlo': 1000,
        'zscore': False}



code_to_name = {'t': 'Probe', 'ct': 'Context'}


########################################################################################################################
def nway_analysis(full_raster, meta):
    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, _, _ = cdPCA.get_centered_means(full_raster)
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
               'rasterfs': load_fs,
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

    full_rast, transitions, contexts = raster_from_sig(sig, cp_regex, goodcells, raster_fs=meta['raster_fs'])

    if len(contexts) < 2:
        real = shuffled = simulated = None
        print(f'only one context for {probe_regex}, skiping analysis')
    else:
        real, shuffled, simulated = nway_analysis(full_rast, meta)

    return real, shuffled, simulated, transitions, contexts


########################################################################################################################

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


# information of what set of sequences pairs to use
duration = 500
source_duration = 500
positions = int(source_duration / _nom2real_dur(duration))
# positions = [0,1]

site = 'AMT028b'  # example site

# prealocating seems too complicated, rather make dicts of lists
all_real = coll.defaultdict(list)
all_shuffled = coll.defaultdict(list)
all_simulated = coll.defaultdict(list)


for source, position in itt.product(range(20), range(positions)):

    probe = NTI_epoch_name(duration, source, position)

    fourway_name = f'191014_{site}_P{probe}_fourway_analysis'
    analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
    analysis_name = 'NTI_LDA_dprime'

    nway = make_cache(function=dPrime_from_NIT_site,
                      func_args={'site': site, 'duration': duration, 'source': source,
                                 'position': position, 'meta': meta},
                      classobj_name=fourway_name,
                      cache_folder=f'/home/mateo/mycache/{analysis_name}/{analysis_parameters}',
                      recache=False)

    real, shuffled, simulated, transitions, contexts = get_cache(nway)  # get_cache(twoway)

    if real is None:
        continue

    trans_pairs = itt.combinations(transitions, 2)
    trans_pairs = [sorted(tp) for tp in trans_pairs] # makes sure to not have repeated pairs as a-b b-a

    for tt, trans_pair in enumerate(trans_pairs):
        pair_name = f'{trans_pair[0]}_{trans_pair[1]}'
        all_real[pair_name].append(real[tt,:])
        all_shuffled[pair_name].append(shuffled[:, tt, :])
        all_simulated[pair_name].append(simulated[:, tt, :])


# stacs all the individual probe instances across a new axis
all_real_arr = {key: np.stack(value, axis=0) for key, value in all_real.items()}
all_shuffled_arr = {key: np.stack(value, axis=0) for key, value in all_shuffled.items()}
all_simulated_arr = {key: np.stack(value, axis=0) for key, value in all_simulated.items()}




# pool across different probes, mean across probe for the real dprime, ignores probe source for the montecarlos
pool_real = {key: np.mean(val, axis=0) for key, val in all_real_arr.items()}
pool_shuffled = {key: np.reshape(val, (val.shape[0] * val.shape[1], val.shape[2]))
                 for key, val in all_shuffled_arr.items()}
pool_simulated = {key: np.reshape(val, (val.shape[0] * val.shape[1], val.shape[2]))
                  for key, val in all_simulated_arr.items()}

####################################################################################################
# pairwise dprime plots
trans_pairs = pool_real.keys()
trans_count = 0

nrow = 1
ncol = len(trans_pairs)
fig, axes = plt.subplots(nrow, ncol, sharey=True, sharex=True, squeeze=False)
for row, col in itt.product(range(nrow), range(ncol)):
    trans_pair = list(trans_pairs)[trans_count]

    real = pool_real[trans_pair]
    shuffled = pool_shuffled[trans_pair]
    simulated = pool_simulated[trans_pair]

    ax = axes[row, col]
    T = real.shape[0]
    time = np.linspace(0, T / meta['raster_fs'], T, endpoint=False)

    ci = 0.9

    # random floor
    # ax.plot(time, np.mean(shuffled[:, :], axis=0), color=ci_color['shuffled'], alpha=1)
    # _cint(time, shuffled, ci, ax=ax,
    #       fillkwargs={'alpha': 0.5, 'color': ci_color['shuffled'], 'label': 'context discrimination'})

    # population effect
    ax.plot(time, np.mean(simulated[:, :], axis=0), color=ci_color['simulated'], alpha=1)
    _cint(time, simulated, ci, ax=ax,
          fillkwargs={'alpha': 0.5, 'color': ci_color['simulated'], 'label': 'population effect'})

    # real dprime
    ax.plot(time, real, color='black', linestyle='-', label='real value')

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

    ax.set_title(f'{trans_pair}', fontsize=sub_title_size)

    # time only in bottom plot
    if row != nrow - 1:
        ax.axes.get_xaxis().set_visible(False)
    else:
        ax.set_xlabel('time (s)', fontsize=ax_lab_size)

    # adds legend in the last ax
    if trans_count + 1 == nrow * ncol:
        ax.legend()

    trans_count += 1

suptitle = f"{site} probe {probe} n-way LDA dprime {meta['raster_fs']}Hz zscore {meta['zscore']}"
fig.suptitle(suptitle, fontsize=sup_title_size)

analysis = f"LDA_n-way_{meta['raster_fs']}Hz_zscore-{meta['zscore']}"

# Export figures
analysis = f"LDA_weights_{meta['raster_fs']}Hz_zscore-{meta['zscore']}"

root = pl.Path(f'/home/mateo/Pictures/APAM/final/{analysis}')
if not root.exists(): root.mkdir(parents=True, exist_ok=True)

png = root.joinpath(suptitle).with_suffix('.png')
fig.savefig(png, transparent=False, dpi=100)

svg = root.joinpath(suptitle).with_suffix('.svg')
fig.savefig(svg, transparent=True)


