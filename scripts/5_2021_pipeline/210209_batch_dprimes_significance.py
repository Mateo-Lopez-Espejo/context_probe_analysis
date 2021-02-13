from src.metrics.significance import _significance
from src.metrics.consolidated_dprimes import single_cell_dprimes, probewise_LDA_dprimes, probewise_dPCA_dprimes, full_dPCA_dprimes
from src.data.load import get_site_ids
import itertools as itt

rec_recache = False
dprime_recache = False

signif_tails = 'both'
alpha=0.01


meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute': None}

analysis_functions = {'SC': single_cell_dprimes,'LDA':probewise_LDA_dprimes,
                      'pdPCA': probewise_dPCA_dprimes, 'fdPCA': full_dPCA_dprimes}


permutations = {'contexts': [0, 1, 2, 3, 4],
                'probes': [1, 2, 3, 4],
                'stim_type': 'permutations'}

triplets = {'contexts': ['silence', 'continuous', 'similar', 'sharp'],
            'probes':[2, 3, 5, 6],
            'stim_type': 'triplets'}

experiments = [permutations, triplets]


sites = set(get_site_ids(316).keys())
badsites = {'AMT031a', 'DRX008b','DRX021a', 'DRX023a', 'ley074a' }  # empirically decided
sites = sites.difference(badsites)
# sites = ['CRD004a']

bads = list()
for site, expt, (fname, func) in itt.product(sites, experiments, analysis_functions.items()):
    print(site, expt['stim_type'], fname)
    expt = expt.copy()
    meta['stim_type'] = expt.pop('stim_type')
    try:
        dprime, shuffled_dprime, goodcells = func(site, **expt, meta=meta)
    except:
        bads.append((site, expt['stim_type'], fname))



# significance, corrected_signif, confidence_interval = _significance(dprime, shuffled_dprime, [1, 2, 3], alpha=alpha)