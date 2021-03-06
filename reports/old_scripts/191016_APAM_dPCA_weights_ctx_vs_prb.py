import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import itertools as itt
import pathlib as pl

import src.data.rasters
from src.data.load import load
from src.metrics.reliability import signal_reliability
from src.data import dPCA as cdPCA

from src.metrics import dprime as cDP

from src.visualization import fancy_plots as cplt

"""
plots the weights over time for the dPCA context marginalization projection, showing a chunk of the preceding context.
The dPCA has been fitted idependently for the context and the probe. 
Aligned to the weights are all the pairwise d' and finally 
the hybrid raster/psth of the most weighted neuron at the time with the highest d' across all d' paris
"""


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def fit_transform(site, probe, meta, part):
    recs = load(site)

    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
    sig = rec['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    raster = src.data.rasters.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                              smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                              part=part, zscore=meta['zscore'])

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    trialR, R = trialR.squeeze(), R.squeeze()  # squeezes out probe

    _, dPCA_projection, _, dpca = cdPCA._cpp_dPCA(R, trialR, significance=False, dPCA_parms={})
    dPCA_projection = dPCA_projection['ct'][:, 0, ...]
    dPCA_weights = np.tile(dpca.D['ct'][:, 0][:, None, None], [1, 1, R.shape[-1]])

    dprime = cDP.pairwise_dprimes(dPCA_projection)

    return dprime, dPCA_projection, dPCA_weights, dpca


def transform(site, probe, meta, part, dpca):
    recs = load(site)

    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')

    rec = recs['trip0']
    sig = rec['resp']

    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    raster = src.data.rasters.raster_from_sig(sig, probe, channels=goodcells, transitions=meta['transitions'],
                                              smooth_window=meta['smoothing_window'], raster_fs=meta['raster_fs'],
                                              part=part, zscore=meta['zscore'])

    # trialR shape: Trial x Cell x Context x Probe x Time; R shape: Cell x Context x Probe x Time
    trialR, R, _ = cdPCA.format_raster(raster)
    trialR, R = trialR.squeeze(), R.squeeze()  # squeezes out probe

    # transforms context trials withe probe fit
    trialZ = cdPCA.transform_trials(dpca, trialR)
    dPCA_projection = trialZ['ct'][:, 0, ...]
    dPCA_weights = np.tile(dpca.D['ct'][:, 0][:, None, None], [1, 1, R.shape[-1]])

    dprime = cDP.pairwise_dprimes(dPCA_projection)

    return dprime, dPCA_projection, dPCA_weights, dpca


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628',  # blue, orange, green, brow,
                  '#984ea3', '#999999', '#e41a1c', '#dede00']  # purple, gray, scarlet, lime

trans_color_map = {'silence': '#377eb8',  # blue
                   'continuous': '#ff7f00',  # orange
                   'similar': '#4daf4a',  # green
                   'sharp': '#a65628'}  # brown

ci_color = {'shuffled': 'orange',
            'simulated': 'purple'}

transitions = {'P2': {'silence': 0,
                      'continuous': 1,
                      'similar': 3,
                      'sharp': 6},
               'P3': {'silence': 0,
                      'continuous': 2,
                      'similar': 1,
                      'sharp': 5},
               'P5': {'silence': 0,
                      'continuous': 4,
                      'similar': 6,
                      'sharp': 3},
               'P6': {'silence': 0,
                      'continuous': 5,
                      'similar': 4,
                      'sharp': 2}}

# transferable plotting parameters
plt.rcParams['svg.fonttype'] = 'none'
sup_title_size = 15
sub_title_size = 12
ax_lab_size = 12
ax_val_size = 11

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'significance': False,
        'montecarlo': 1000,
        'zscore': False,
        'same_trans': False}

analysis_name = 'dPCA_weights'
analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
code_to_name = {'t': 'Probe', 'ct': 'Context'}

all_probes = [2, 3, 5, 6]
all_sites = ['ley070a',  # good site. A1
             'ley072b',  # Primary looking responses with strong contextual effects
             'AMT028b',  # good site
             'AMT029a',  # Strong response, somehow visible contextual effects
             'AMT030a',  # low responses, Ok but not as good
             # 'AMT031a', # low response, bad
             'AMT032a']  # great site. PEG

# all_sites = ['AMT029a']
# all_probes = [5]
for site, probe in zip(['AMT029a', 'ley070a'], [5, 2]):
    # for site, probe in itt.product(all_sites, all_probes):

    # gets signal for hybridplot and toe select goodcellss
    recs = load(site)
    if len(recs) > 2:
        print(f'\n\n{recs.keys()}\n\n')
    rec = recs['trip0']
    sig = rec['resp']
    # calculates response realiability and select only good cells to improve analysis
    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # fits and concatenates context probe analysis along the time axis
    dprimesL = list();
    projectionsL = list();
    weightsL = list()

    if meta['same_trans'] is False:
        for part in ['context', 'probe']:
            try:
                badflag = False
                d, p, w, dpca = fit_transform(site, probe, meta, part)
                dprimesL.append(d);
                projectionsL.append(p);
                weightsL.append(w)
            except:
                print(f'failed to analize {site} probe{probe}')
                badflag = True
                break

    elif meta['same_trans'] is True:
        try:
            badflag = False
            # fits transfomrs probe response, append to list
            d, p, w, dpca = fit_transform(site, probe, meta, 'probe')
            dprimesL.append(d);
            projectionsL.append(p);
            weightsL.append(w)

            # transforms  context response, prepends to list
            d, p, w, dpca = transform(site, probe, meta, 'context', dpca)
            dprimesL.insert(0, d);
            projectionsL.insert(0, p);
            weightsL.insert(0, w)

        except:
            print(f'failed to analize {site} probe{probe}')
            badflag = True

    if badflag: continue

    half = dprimesL[0].shape[1]
    dprimes = np.concatenate(dprimesL, axis=-1)
    projections = np.concatenate(projectionsL, axis=-1)
    weights = np.concatenate(weightsL, axis=-1)

    # cuts the whole lenght of the recording to a defined time length
    start = 0.8
    end = 1.5
    offset = start - 1
    stbin = int(np.floor(start * meta['raster_fs']))
    enbin = int(np.floor(end * meta['raster_fs']))
    osbin = int(np.floor(offset * meta['raster_fs']))

    dprimes = dprimes[..., stbin:enbin]
    projections = projections[..., stbin:enbin]
    weights = weights[..., stbin:enbin]

    half = -osbin

    # defines time bins ins seconds
    t = (np.arange(0, dprimes.shape[1]) / meta['raster_fs']) + offset

    # finds highest dprime during probe
    topDbin = np.where(dprimes == np.max(dprimes[:, half:]))[1][0]
    topDtime = topDbin / meta['raster_fs'] + offset

    # formats weights to have consistent signs with the top dprime bin as reference
    toflip = (np.dot(weights[:, 0, :].T, weights[:, 0, topDbin]) < 0) * -2 + 1
    fweights = weights * toflip[None, None, :]

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes = np.ravel(axes)

    # transformation weights
    trans_ax = axes[0]
    wmax = np.max(np.abs(fweights))
    wmin = -wmax
    trans_im = trans_ax.imshow(fweights.squeeze(), aspect='auto',
                               extent=[offset, end + offset - start, fweights.shape[0], 0],
                               cmap='PuOr', clim=(wmin, wmax), norm=MidpointNormalize(midpoint=0, vmin=wmin, vmax=wmax))
    trans_ax.axvline(0, linestyle=':', color='gray')
    trans_ax.axvline(topDtime, linestyle='--', color='Black')

    cbar_ax = inset_axes(trans_ax,
                         width="1%",  # width = 50% of parent_bbox width
                         height="50%",  # height : 5%
                         loc='upper right')

    fig.colorbar(trans_im, cax=cbar_ax, orientation='vertical', ticks=[wmin, 0, wmax])
    cbar_ax.yaxis.set_ticks_position("left")

    # dprimes
    dprime_ax = axes[1]
    for ii, (c0, c1) in enumerate(itt.combinations(meta['transitions'], 2)):
        dprime_ax.plot(t, dprimes[ii, :], label=f"{c0} vs {c1}")
    dprime_ax.legend()
    dprime_ax.axvline(0, linestyle=':', color='gray')
    dprime_ax.axvline(topDtime, linestyle='--', color='black')

    # raw raster
    epoch_names = [f"C{transitions[f'P{probe}'][trans]}_P{probe}" for trans in meta['transitions']]
    topcell_idx = np.argmax(np.abs(weights[:, 0, topDbin]))
    topcell = goodcells[topcell_idx]
    trans_colors = [trans_color_map[trans] for trans in meta['transitions']]

    raster_ax = axes[2]
    cplt.hybrid(sig, epoch_names, channels=topcell, psth_fs=meta['raster_fs'],
                time_strech=[start, end], time_offset=offset, axes=[raster_ax],
                legend=True, labels=meta['transitions'], colors=trans_colors)
    raster_ax.axvline(0, linestyle=':', color='gray')
    raster_ax.axvline(topDtime, linestyle='--', color='black')

    # horizonta line in weightplot indicating most weighted cell
    trans_ax.axhline(topcell_idx + 0.5, linestyle='--', color='Black')

    # Formatting

    for ax in [trans_ax, dprime_ax, raster_ax]:
        ax.tick_params(labelsize=ax_val_size)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    # ax labels
    trans_ax.set_ylabel('neuron\nweights', fontsize=ax_lab_size)
    dprime_ax.set_ylabel("pairwise\nd'", fontsize=ax_lab_size)

    raster_ax.set_ylabel('top cell\nspike rate (Hz)', fontsize=ax_lab_size)
    raster_ax.set_xlabel('time (s)', fontsize=ax_lab_size)

    raster_ax.set_title('')

    # suptitle = f"{site} probe {probe} dPCA weights {meta['raster_fs']}Hz zscore {meta['zscore']} same_trans-{meta['same_trans']}"
    suptitle = f"{site} probe {probe} dPCA"
    fig.suptitle(suptitle, fontsize=20)
    fig.set_size_inches(7, 4)
    # fig.set_size_inches(14, 7)

    # Export figures
    analysis = f"dPCA_weights_{meta['raster_fs']}Hz_zscore-{meta['zscore']}_same_trans-{meta['same_trans']}"

    root = pl.Path(f'/home/mateo/Pictures/APAM/final/{analysis}')
    if not root.exists(): root.mkdir(parents=True, exist_ok=True)

    png = root.joinpath(suptitle).with_suffix('.png')
    fig.savefig(png, transparent=False, dpi=100)

    svg = root.joinpath(suptitle).with_suffix('.svg')
    fig.savefig(svg, transparent=True)
