import itertools as itt
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import curve_fit

import cpn_LDA as cLDA
import cpn_dPCA as cdPCA
import cpn_dprime as cDP
from cpn_load import load, get_site_ids
from cpp_cache import make_cache, get_cache
from reliability import signal_reliability
from tools import shuffle_along_axis as shuffle

'''
since applying the dprime CPN analysis toe the NTI data was unsuccessfull, the next alternative to compare Sam and my
approach is to perform the CPN and NTI analysis to their respective datasets on recording sites that have both data
'''


# 1. list sites with both datasets
# list all NTI sites this have to be done manually
# list all CPN sites, this should be trivial
# check the intersection

# 2. Calculates the dPrime for each site and all possible probes, context pairs and cells (?). This is the difficult part
# to summarize the outcome of all the

def exp(x, a, b):
    return a * np.exp(b * x)


def fit_exp_decay(times, values):
    """
    fits a properly constrained exponential decay to the times and values give, retursn the fitted values
    of the exponential function and the equivalent time constant Tau
    :param times: np.array. 1D, Time points in seconds, same shape as values
    :param values: np.array. 1D, y values, same shape as times
    :return:
    """
    popt, pvar = curve_fit(exp, times, values, p0=[1, 0], bounds=([0, -np.inf], [np.inf, 0]))
    return popt, pvar


def plot_exp_decay(times, values, ax=None, label=True, pltkwargs={}):
    '''
    plots an exponential decaye curve fited on times and values
    :param times:
    :param values:
    :param ax:
    :param label:
    :param pltkwargs:
    :return:
    '''
    defaults = {'color': 'gray', 'linestyle': '--'}
    defaults.update(**pltkwargs)

    popt, pvar = fit_exp_decay(times, values)

    if ax == None:
        fig, ax = plt.subplots()
    else:
        ax = ax
        fig = ax.get_figure()

    if label == True:
        label = 'start={:+.2f}, tau= {:+.2f}'.format(popt[0], -1 / popt[1])
    elif label == False:
        label = None
    else:
        pass

    ax.plot(times, exp(times, *popt), color='gray', linestyle='--',
            label=label)

    return fig, ax, popt, pvar


def cell_dprime(site, probe, meta):
    recs = load(site)
    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
    sig = rec['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = cdPCA.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                   smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                   zscore=meta['zscore'], part='probe')

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    trialR, R = trialR.squeeze(), R.squeeze()  # squeezes out probe

    rep, chn, ctx, tme = trialR.shape

    trans_pairs = [f'{x}_{y}' for x, y in itt.combinations(meta['transitions'], 2)]

    dprime = cDP.pairwise_dprimes(trialR, observation_axis=0, condition_axis=2)  # shape CellPair x Cell x Time

    # Shuffles the rasters n times and organizes in an array with the same shape the raster plus one dimension
    # with size n containing each shuffle

    shuffled = list()
    # pbar = ProgressBar()
    print(f"\nshuffling {meta['montecarlo']} times")
    for tp in trans_pairs:
        shuf_trialR = np.empty([meta['montecarlo'], rep, chn, 2, tme])
        shuf_trialR[:] = np.nan

        tran_idx = np.array([meta['transitions'].index(t) for t in tp.split('_')])
        ctx_shuffle = trialR[:, :, tran_idx, :].copy()

        for rr in range(meta['montecarlo']):
            shuf_trialR[rr, ...] = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)

        shuffled.append(cDP.pairwise_dprimes(shuf_trialR, observation_axis=1, condition_axis=3))

    shuffled = np.stack(shuffled, axis=1)  # shape Montecarlo x ContextPair x Cell x Time

    return dprime, shuffled, goodcells, trans_pairs


def dPCA_fourway_analysis(site, probe, meta):
    recs = load(site)

    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
    sig = rec['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # get the full data raster Context x Probe x Rep x Neuron x Time
    raster = cdPCA.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                   smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                   zscore=meta['zscore'])

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    trialR, R = trialR.squeeze(), R.squeeze()  # squeezes out probe
    Re, C, S, T = trialR.shape

    # calculates full dPCA. i.e. considering all 4 categories
    def fit_transformt(R, trialR):
        _, dPCA_projection, _, dpca = cdPCA.trials_dpca(R, trialR, significance=False, dPCA_parms={})
        dPCA_projection = dPCA_projection['ct'][:, 0, ]
        dPCA_transformation = np.tile(dpca.D['ct'][:, 0][:, None, None], [1, 1, T])
        return dPCA_projection, dPCA_transformation

    dPCA_projection, dPCA_transformation = fit_transformt(R, trialR)
    dprime = cDP.pairwise_dprimes(dPCA_projection)

    # calculates floor (ctx shuffle) and ceiling (simulated data)
    sim_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))
    shuf_dprime = np.empty([meta['montecarlo']] + list(dprime.shape))

    ctx_shuffle = trialR.copy()

    # pbar = ProgressBar()
    for rr in range(meta['montecarlo']):
        # ceiling: simulates data, calculates dprimes
        sim_trial = np.random.normal(np.mean(trialR, axis=0), np.std(trialR, axis=0),
                                     size=[Re, C, S, T])
        sim_projection = cLDA.transform_over_time(cLDA._reorder_dims(sim_trial), dPCA_transformation)
        sim_dprime[rr, ...] = cDP.pairwise_dprimes(cLDA._recover_dims(sim_projection).squeeze())

        ctx_shuffle = shuffle(ctx_shuffle, shuffle_axis=2, indie_axis=0)
        shuf_projection = cLDA.transform_over_time(cLDA._reorder_dims(ctx_shuffle), dPCA_transformation)
        shuf_dprime[rr, ...] = cDP.pairwise_dprimes(cLDA._recover_dims(shuf_projection).squeeze())

    return dprime, shuf_dprime, sim_dprime

# transferable plotting parameters
plt.rcParams['svg.fonttype'] = 'none'
sup_title_size = 30
sub_title_size = 20
ax_lab_size = 15
ax_val_size = 11

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'montecarlo': 1000,
        'zscore': False}

analysis_name = 'NTI_singel_cell_dprime'
analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
code_to_name = {'t': 'Probe', 'ct': 'Context'}

all_probes = [2, 3, 5, 6]

sites = ['ley070a',  # good site. A1
         'ley072b',  # Primary looking responses with strong contextual effects
         'AMT028b',  # good site
         'AMT029a',  # Strong response, somehow visible contextual effects
         'AMT030a',  # low responses, Ok but not as good
         # 'AMT031a', # low response, bad
         'AMT032a']  # great site. PEG

# sites = list(get_site_ids(316).keys())
# problem sites:
# sites = ['AMT031a']


# for site, probe in zip(['AMT029a', 'ley070a'],[5,2]):
# all_sites = ['AMT029a']
# all_sites = ['AMT032a']
# all_probes = [5]

bad_sites = list()
all_pvalues = dict()
all_reals = dict()
all_shuffled = dict()

for site in sites:

    this_site_reals = list()
    this_site_shuffled = list()
    this_site_pvalues = list()
    for pp, probe in enumerate(all_probes):
        object_name = f'200221_{site}_P{probe}_single_cell_dprime'
        # cache_folder = pl.Path('U:\\mateo' , 'mychache' , analysis_name , analysis_parameters)
        cache_folder = pl.Path('C:\\', 'users', 'mateo', 'mycache', analysis_name, analysis_parameters)

        try:
            cache = make_cache(function=cell_dprime,
                               func_args={'site': site, 'probe': probe, 'meta': meta},
                               classobj_name=object_name,
                               cache_folder=cache_folder,
                               recache=False)

            real, shuffled, cell_names, trans_pairs = get_cache(cache)

        except:
            bad_sites.append(f"{site}_P{probe}_dPCA")
            continue

        this_site_reals.append(real)
        this_site_shuffled.append(shuffled)

        # single tailed p value base on the montecarlo shuffling

        pvalues = np.sum((shuffled >= real), axis=0) / meta['montecarlo']

        this_site_pvalues.append(pvalues)

    this_site_reals = np.stack(this_site_reals, axis=0)
    this_site_shuffled = np.stack(this_site_shuffled, axis=0)
    this_site_pvalues = np.stack(this_site_pvalues, axis=0)

    # reorders date in dictionary of cells
    for cc, cell in enumerate(cell_names):
        all_reals[cell] = this_site_reals[:, :, cc, :]
        all_shuffled[cell] = this_site_shuffled[:, :, :, cc, :].swapaxes(0, 1)
        all_pvalues[cell] = this_site_pvalues[:, :, cc, :]

# stacks the site individual arrays along a new site dimension. since the sites have disimilar cell number, pads
all_cells = np.array(list(all_pvalues.keys()))

threshold = 0.05
all_signif = {key: (val <= threshold) for key, val in all_pvalues.items()}
sig_array = np.stack(list(all_signif.values()), axis=0)  # dimensions: Cell x Probe x trans_pair x time

# calculates exponential decay for each cell, collapsing across all probes and transisions
nbin = sig_array.shape[-1]
fs = meta['raster_fs']
times = np.linspace(0, nbin / fs, nbin, endpoint=False) * 1000 # units in ms!!!!
collapsed = sig_array.mean(axis=(1, 2))

# organizes in a dataframe with columns r0: y intercept, decay: eponential valaue and tau: Time to a 36% amplitude
df = list()
for cellid, data in zip(all_cells, collapsed):
    popt, _ = fit_exp_decay(times, data)
    df.append({'cellid': cellid,
               'r0_au': popt[0],
               'decay_ms': popt[1]})

context_fits = pd.DataFrame(df)
context_fits['tau_ms'] = -1/context_fits['decay_ms']
context_fits.set_index(['cellid'], inplace=True)


# 3. import and parse matlab results for Sam's NTI analysis. These results are in a cell by cell format, then it makes
# sense to calculate the dprimes idividually forP each cell
file = pl.Path('C:\\', 'Users', 'Mateo', 'Documents', 'Science', 'code', 'integration_quilt', 'scrambling-ferrets',
               'analysis', 'model_fit_pop_summary').with_suffix('.mat')

best_fits = loadmat(file)['best_fits'].squeeze()
# orders the data in DF
df = list()
for row in best_fits:
    df.append({'cellid': row[2][0],
               'intper_ms': row[0][0][0],
               'delay_ms': row[1][0][0]})

integration_fits = pd.DataFrame(df)
integration_fits.set_index(['cellid'], inplace=True)


# 4. pools together both approache, selects only common cell, plots relationships
# join='inner' keeps only the intersection between the two Dfs, i.e. the cells that have both approaches
DF = pd.concat([context_fits, integration_fits], axis=1, join='inner')

DF.plot(x='tau_ms', y='intper_ms', kind='scatter')