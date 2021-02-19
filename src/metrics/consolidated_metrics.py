import numpy as np
import numpy.ma as ma

from src.metrics.significance import _significance
from src.metrics.consolidated_dprimes import single_cell_dprimes

"""
takes all the dprimes and pvalues, fits exponential decays to both the dprimes and the profiles of dprime
significance (derived from pvalues). 
This is done for all combinations of probe, and context transtion pairs.
This is done for single cells (SC), probewise dPCA (pdPCA) and full_dPCA (fdPCA).
"""


rec_recache = False
dprime_recache = False
signif_tails = 'both'
alpha=0.01

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None,
        'stim_type': 'permutations'}

permutations = {'contexts': [0, 1, 2, 3, 4],
                'probes': [1, 2, 3, 4],}

dprimes, shuffled_dprimes, goodcells = single_cell_dprimes('CRD004a', **permutations, meta=meta)



# calculate significant time bins, both raw and corrected for multiple comparisons
significance, corrected_signif, confidence_interval = _significance(dprimes, shuffled_dprimes, [1,2,3], alpha=0.01)


# masks dprime with different significances
dprimes_mask = ma.masked_array(dprimes, ~significance)

# take multiple means across transision pairs, probes or both
def _get_multiple_means(dprimes):
    """
    Private function to get the different combinations of means across probes, contexts pairs, or both
    :param dprime: ndarray with shape Channel x ContextPairs x Probes x Time
    :return:
    """
    mean_dicts = dict()
    # mean of transition pairs for each probe
    mean_dicts['probe'] = np.mean(dprimes, axis=1)
    mean_dicts['transition_pair'] = np.mean(dprimes, axis=2)
    mean_dicts['full'] = np.mean(dprimes, axis=(1,2))

    return mean_dicts

means = _get_multiple_means(dprimes_mask)


# defines all metrics functions with homogeneous input and output shapes






# helper function to transform "labeled" ndimensional arrays into flat, long format pandas dataframes

# import timeit
# number=1
# print(timeit.timeit('o0(arr, all_lab)', globals=globals(), number=number))
# print(timeit.timeit('o1(arr, all_lab)', globals=globals(), number=number))
# print(timeit.timeit('o2(arr, all_lab)', globals=globals(), number=number))

# config = ConfigParser()
# config.read_file(open(pl.Path(__file__).parents[2] / 'config' / 'settings.ini'))
#
# meta = {'reliability': 0.1,  # r value
#         'smoothing_window': 0,  # ms
#         'raster_fs': 30,
#         'transitions': [0, 1, 2, 3, 4],
#         'montecarlo': 1000,
#         'zscore': True,
#         'dprime_absolute': None}
#
# recache = True
#
# # loads the raw calculated dprimes and montecarlos
# batch_dprime_file = pl.Path(config['paths']['analysis_cache']) / 'prm_dprimes_v2' / set_name(meta)
# batch_dprimes = jl.load(batch_dprime_file)
#
# # defines significant values based on loaded pvalue and defined threshold
# threshold = 0.01
# for analysis_name, mid_dict in batch_dprimes.items():
#     mid_dict['significance'] = {key: (val <= threshold) for key, val in mid_dict['pvalue'].items()}
#
# # set up the time bin labels in milliseconds, this is critical for plotting and calculating the tau
# nbin = np.max([value.shape[-1] for value in batch_dprimes['SC']['dprime'].values()])
# times = np.linspace(0, nbin / meta['raster_fs'], nbin, endpoint=False) * 1000
#
# sites = set(batch_dprimes['pdPCA']['dprime'].keys())
# all_probes = [1, 2, 3, 4]
# all_trans = [f'{t0}_{t1}' for t0, t1 in itt.combinations(meta['transitions'], 2)]
#
# # creates and caches, or loads the DF
#
# summary_DF_file = pl.Path(config['paths']['analysis_cache']) / 'prm_summary_DF_v2' / set_name(meta)