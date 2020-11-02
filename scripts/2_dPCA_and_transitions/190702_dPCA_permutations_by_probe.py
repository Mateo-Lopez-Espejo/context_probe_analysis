import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl

from dPCA import dPCA

import src.visualization.fancy_plots
from src.data.load import load
from src.metrics.reliability import signal_reliability
from src.data import dPCA as cdPCA, triplets as tp

all_sites = ['ley070a', # good site. A1
             'ley072b', # Primary looking responses with strong contextual effects
             'AMT028b', # good site
             'AMT029a', # Strong response, somehow visible contextual effects
             'AMT030a', # low responses, Ok but not as good
             # 'AMT031a', # low response, bad
             'AMT032a'] # great site. PEG

meta = {'reliability' : 0.1, # r value
        'smoothing_window' : 20, # ms
        'probes_to_plot': [1,2,3,4]}


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

    fig = plt.figure()

    for pp, probe in enumerate(meta['probes_to_plot']):

        # get a specific probe after a set of different transitions
        trialR = full_array[:, probe, :, :, 100:][:, None, :, :, :]  # only includes response to probe
        # reorders dimentions from Context x Probe x Trial x Neuron x Time  to  Trial x Neuron x Context x Probe x Time
        trialR, R, _ = cdPCA.format_raster(trialR)
        Tr, N, C, P, T = trialR.shape

        trialR, R = trialR.squeeze(), R.squeeze()

        n_components = N if N < 10 else 10

        # initializes model
        dpca = dPCA.dPCA(labels='ct',regularizer='auto', n_components=n_components,
                         join={'ct' : ['c','ct']})
        dpca.protect = ['t']

        # Now fit the data (R) using the model we just instantiated. Note that we only need trial-to-trial data when we want to
        # optimize over the regularization parameter.2
        Z = dpca.fit_transform(R,trialR)

        # check for significance FixME
        # significance_masks = dpca.significance_analysis(R,trialR, axis='t',n_shuffles=10,n_splits=2,n_consecutive=1)

        # plots projection in first component for each marginalization, probe and context

        time = np.arange(T)
        # plots probe response given different contexts(color), marignalization (subplot column), and probe (subplot row)
        for (mm, marginalization) in enumerate(dpca.marginalizations.keys()):

            # get the axis
            ax = plt.subplot2grid((4,4), (pp, mm), rowspan=1, colspan=1, fig=fig)

            for cc, context in enumerate(all_contexts):
                toplot = Z[marginalization][0, cc, :]
                ax.plot(toplot, color=f'C{cc}', label=context)

            # set labels
            title = marginalization if pp == 0 else None
            ylab = f'{probe} norm spk rate' if mm ==0 else None
            xlab = 'time (s)' if pp == 3 else None

            ax.set_title(title)
            ax.set_ylabel(ylab)
            ax.set_xlabel(xlab)

            if pp==0 and mm == 1: ax.legend()

            # plots explained varianceP
            var_ax = plt.subplot2grid((4,4), (pp, 3), rowspan=1, colspan=1, fig=fig)
            src.visualization.fancy_plots.variance_explained(dpca, var_ax)

            # marginalization weights
            weight_ax = plt.subplot2grid((4,4), (pp, 2), rowspan=1, colspan=1, fig=fig)
            src.visualization.fancy_plots.weight_pdf(dpca, marginalization='ct', axes=weight_ax, cellnames=goodcells)

            fig.suptitle(f'{site},all cp permutations')

        # set figure to full size in tenrec screen
        fig.set_size_inches(19.2, 9.79)

        analysis_parameters = '_'.join(['{}-{}'.format(key, str(val)) for key, val in meta.items()])
        root = pl.Path(f'/home/mateo/Pictures/perm_probewise_dPCA/{analysis_parameters}')
        if not root.exists(): root.mkdir(parents=True, exist_ok=True)
        filepath = root.joinpath(site).with_suffix('.png')
        fig.savefig(filepath, dpi=100)


