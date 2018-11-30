import collections as col
import itertools as itt

import cpp_cache as ccache
import cpp_dispersion as cdisp
import cpp_epochs as cep
import cpp_reconstitute_rec as crec

batch = 310

all_models = ['wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1', 'wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1',
              'wc.2x2.c-fir.2x15-lvl.1-stategain.S-dexp.1', 'wc.2x2.c-fir.2x15-lvl.1-dexp.1']

shortnames = {'resp': 'resp',
              'wc.2x2.c-fir.2x15-lvl.1-dexp.1': 'LN',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1': 'STP',
              'wc.2x2.c-fir.2x15-lvl.1-stategain.S-dexp.1': 'state',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1': 'STP_state'}

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

# renames for ease
# renamed = {site: {shortnames[model]: rec for model, rec in recs.items() } for site, recs in pop_recs.items()}

# iterates over each site id, and calculates the dispersion for predictions of each different model

site_disps = dict()

for site, recs in pop_recs.items():
    # for multi core running
    # site_IDs = list(pop_recs.keys())
    # site = site_IDs[4]
    # recs = pop_recs[site]

    formated = {key: cep.set_recording_subepochs(rec) for key, rec in recs.items()}

    # calculatest the dispersion for each prediction and one response
    dispersions = {modelname: cdisp.signal_all_context_sigdif(rec['pred'], channels='all',
                                                              signal_name='{}_{}'.format(modelname, site),
                                                              probes=(1, 2, 3, 4), dimensions='population', sign_fs=100,
                                                              window=1, rolling=True, type='Euclidean', recache=False,
                                                              value='metric')[0]
                   for modelname, rec in formated.items()}

    real_neu_resp = formated['wc.2x2.c-fir.2x15-lvl.1-dexp.1']['resp']
    dispersions['resp'] = cdisp.signal_all_context_sigdif(real_neu_resp, channels='all',
                                                          signal_name='{}_{}'.format('resp', site),
                                                          probes=(1, 2, 3, 4), dimensions='population', sign_fs=100,
                                                          window=1, rolling=True, type='Euclidean', recache=False,
                                                          value='metric')[0]


    site_disps[site] = dispersions

for site, disps in site_disps.items():


