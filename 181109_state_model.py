import logging
import nems.modelspec as ms
import nems.xforms as xforms
import nems_db.xform_wrappers as nw
import joblib as jl
# todo elimiate these import later
import matplotlib.pyplot as plt
import numpy as np
import nems_lbhb
import nems_db.db as nd

# from nems.gui.recording_browser import browse_recording, browse_context
log = logging.getLogger(__name__)

batch = 310
cellid = 'BRT037b-39-1'
loadkey = "env.fs100.cst"

options = {'stimfmt': 'envelope',
           'rasterfs': 100,
           'recache': True,
           'runclass': 'CPP',
           'stim': True}

outpath = '/tmp/'

ctx = nw.baphy_load_wrapper(cellid=cellid, batch=batch,
                            loadkey=loadkey, options=options)

ctx.update(xforms.load_recordings(
    cellid=cellid, save_other_cells_to_state=True, **ctx))

# if one stim is repeated a lot, can use it as val
ctx.update(xforms.split_by_occurrence_counts(
   epoch_regex='^STIM_', **ctx))

# ctx.update(xforms.split_at_time(fraction=0.8, **ctx))
# uncommenting will only allow signal correlations to help
# ctx.update(xforms.average_away_stim_occurrences(**ctx)) # there is a bug downstream
modelspec_name = 'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1'

# record some meta data for display and saving
meta = {'cellid': cellid, 'batch': batch,
        'modelname': modelspec_name, 'recording': cellid}

ctx.update(xforms.init_from_keywords(modelspec_name, meta=meta, **ctx))

# temp cache until bugs post fit are solved
refit = True
if refit == True:
    ctx.update(xforms.fit_basic_init(**ctx))
    ctx.update(xforms.fit_basic(**ctx))
    jl.dump(ctx, '/home/mateo/code/context_probe_analysis/pickles/ctx_fit_all')
elif refit == False:
    full_ctx = jl.load('/home/mateo/code/context_probe_analysis/pickles/ctx_fit_all')
    ctx = full_ctx
else:
    raise ValueError('refit must be bool')

# checking integrity of contexts
# resp = ctx['rec']['resp'].rasterize()._data
# est = ctx['est']['resp']._data
# val = ctx['val']['resp']._data
# plt.figure()
# plt.plot(resp.T)
# plt.plot(est.T + 3)
# plt.plot(val.T + 3)
# # stacks est and val and scales to proper time
# merge = np.concatenate([est, val], axis=0)
# merge = np.nansum(merge,axis=0)
# #plt.plot(merge+6)
# gaps = np.all(np.concatenate([np.isnan(est), np.isnan(val)], axis=0), axis=0)
# plt.plot(7*gaps-1)


ctx.update(xforms.predict(**ctx))

ctx.update(xforms.add_summary_statistics(**ctx))

ctx.update(xforms.plot_summary(**ctx))



# # save results
# modelspecs = ctx['modelspecs']
#
# destination = '/auto/data/nems_db/results/{0}/{1}/{2}/'.format(
#     batch, cellid, ms.get_modelspec_longname(modelspecs[0]))
# modelspecs[0][0]['meta']['modelpath'] = destination
# modelspecs[0][0]['meta']['figurefile'] = destination + 'figure.0000.png'
# modelspecs[0][0]['meta'].update(meta)
# xforms.save_analysis(destination,
#                      recording=ctx['rec'],
#                      modelspecs=modelspecs,
#                      xfspec=xfspec,# from sklearn.decomposition import PCA
#                      figures=ctx['figures'],
#                      log=log_xf)