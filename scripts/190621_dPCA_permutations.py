import matplotlib.pyplot as plt
import numpy as np
import itertools as itt
import pathlib as pl

from dPCA import dPCA

import cpn_triplets as tp
from cpn_load import load
from cpn_reliability import signal_reliability
import cpn_dPCA as cdPCA


all_sites = ['ley070a', # good site. A1
             'ley072b', # Primary looking responses with strong contextual effects
             'AMT028b', # good site
             'AMT029a', # Strong response, somehow visible contextual effects
             'AMT030a', # low responses, Ok but not as good
             # 'AMT031a', # low response, bad
             'AMT032a'] # great site. PEG

meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 20} # ms



for site in all_sites:
    # load and format triplets from a site
    recs = load(site)
    rec = recs['perm0']
    sig = rec['resp'].rasterize()

    # calculates response realiability and select only good cells to improve analysis

    r_vals, goodcells = signal_reliability(sig, r'\ASTIM_*', threshold=meta['reliability'])
    goodcells = goodcells.tolist()

    # plots PSTHs of all probes after silence
    # fig, axes = cplot.hybrid(sig, epoch_names=r'\AC0_P\d\Z', channels=goodcells)

    # plots PSHTs of individual best probe after all contexts
    # fig, axes = cplot.hybrid(sig, epoch_names=r'\AC\d_P3\Z', channels=goodcells)

    # takes an example probe
    full_array, invalid_cp, valid_cp, all_contexts, all_probes = \
        tp.make_full_array(sig, channels=goodcells, smooth_window=meta['smoothing_window'])


    # get a specific probe after a set of different transitions

    trialR = full_array[:, 1:, :, :, 100:] # excludes silence as context, only includes response to probe
    all_probes.pop(0)

    # reorders dimentions from Context x Probe x Trial x Neuron x Time  to  Trial x Neuron x Context x Probe x Time
    trialR, R, _= cdPCA.format_raster(trialR)
    Tr, N, C, P, T = trialR.shape

    n_components = N if N < 10 else 10

    # initializes model
    dpca = dPCA.dPCA(labels='cpt',regularizer='auto', n_components=n_components,
                     join={'ct' : ['c','ct'], 'pt':['p', 'pt'], 'cpt':['cp', 'cpt']})
    dpca.protect = ['t']

    # Now fit the data (R) using the model we just instantiated. Note that we only need trial-to-trial data when we want to
    # optimize over the regularization parameter.2
    Z = dpca.fit_transform(R,trialR)

    # check for significance FixME
    # significance_masks = dpca.significance_analysis(R,trialR, axis='t',n_shuffles=10,n_splits=2,n_consecutive=1)

    # plots projection in first component for each marginalization, probe and context

    time = np.arange(T)
    fig = plt.figure()

    # plots probe response given different contexts(color), marignalization (subplot column), and probe (subplot row)
    for (pp, probe), (mm, marginalization) in itt.product(enumerate(all_probes), enumerate(dpca.marginalizations.keys())):

        # get the axis
        ax = plt.subplot2grid((5,5), (pp, mm), rowspan=1, colspan=1, fig=fig)

        for cc, context in enumerate(all_contexts):
            toplot = Z[marginalization][0, cc, pp, :]
            ax.plot(toplot, color=f'C{cc}', label=context)

        # set labels
        title = marginalization if pp == 0 else None
        ylab = f'{probe} norm spk rate' if mm ==0 else None
        xlab = 'time (s)' if pp == 3 else None

        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)

        if pp==0 and mm == 3: ax.legend()

    # plots explained varianceP
    var_ax = plt.subplot2grid((5,5), (0, 4), rowspan=5, colspan=1, fig=fig)
    cdPCA.variance_explained(dpca, var_ax)

    # marginalization weights
    # 1. creates a list of axes
    weight_axes = [plt.subplot2grid((5,5), (4, cc), rowspan=1, colspan=1, fig=fig)
                   for cc in range(len(dpca.marginalizations.keys()))]

    # passes to a convenient function
    cdPCA.weight_pdf(dpca, axes=weight_axes, cellnames=goodcells)

    fig.suptitle(f'{site},all cp permutations')

    # set figure to full size in tenrec screen
    fig.set_size_inches(19.2, 9.79)

    analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
    root = pl.Path(f'/home/mateo/Pictures/permutations_dPCA/{analysis_parameters}')
    if not root.exists(): root.mkdir(parents=True, exist_ok=True)
    filepath = root.joinpath(site).with_suffix('.png')
    fig.savefig(filepath, dpi=100)



