import logging
import nems.modelspec as ms
import nems.xforms as xforms
import nems_db.xform_wrappers as nw

# from nems.gui.recording_browser import browse_recording, browse_context
log = logging.getLogger(__name__)

batch = 310
cellid = 'BRT037b-06-1'
loadkey = "env.fs100.cst"

options = {'stimfmt': 'envelope',
           'rasterfs': 100,
           'recache': False,
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
ctx.update(xforms.average_away_stim_occurrences(**ctx))

modelspec_name = 'wc.2x2.c-fir.2x15-lvl.1-stategain.S'

# record some meta data for display and saving
meta = {'cellid': cellid, 'batch': batch,
        'modelname': modelspec_name, 'recording': cellid}

ctx.update(xforms.init_from_keywords(modelspec_name, meta=meta, **ctx))

ctx.update(xforms.fit_basic_init(**ctx))
ctx.update(xforms.fit_basic(**ctx))

ctx.update(xforms.predict(**ctx))

ctx.update(xforms.add_summary_statistics(**ctx))

ctx.update(xforms.plot_summary(**ctx))

# save results
# log.info('Saving modelspec(s) to {0} ...'.format(destination))
modelspecs = ctx['modelspecs']

destination = '/auto/data/nems_db/results/{0}/{1}/{2}/'.format(
    batch, cellid, ms.get_modelspec_longname(modelspecs[0]))
modelspecs[0][0]['meta']['modelpath'] = destination
modelspecs[0][0]['meta']['figurefile'] = destination + 'figure.0000.png'
modelspecs[0][0]['meta'].update(meta)
#
# xforms.save_analysis(destination,
#                      recording=ctx['rec'],
#                      modelspecs=modelspecs,
#                      xfspec=xfspec,# from sklearn.decomposition import PCA
#                      figures=ctx['figures'],
#                      log=log_xf)
# TODO : db results finalized?
# nd.update_results_table(modelspecs[0])


"""
xfspec.append(['nems.xforms.average_away_stim_occurrences', {}])

# MODEL SPEC
# modelspecname = 'dlog_wcg18x1_stp1_fir1x15_lvl1_dexp1'
modelspecname = 'wc.18x1.g_fir.1x15_lvl.1'

meta = {'cellid': 'TAR010c-18-1', 'batch': 271, 'modelname': modelspecname}

xfspec.append(['nems.xforms.init_from_keywords',
               {'keywordstring': modelspecname, 'meta': meta}])

xfspec.append(['nems.xforms.fit_basic_init', {}])
xfspec.append(['nems.xforms.fit_basic', {}])
# xfspec.append(['nems.xforms.fit_basic_shrink', {}])
#xfspec.append(['nems.xforms.fit_basic_cd', {}])
# xfspec.append(['nems.xforms.fit_iteratively', {}])
xfspec.append(['nems.xforms.predict',    {}])
# xfspec.append(['nems.xforms.add_summary_statistics',    {}])
xfspec.append(['nems.analysis.api.standard_correlation', {},
               ['est', 'val', 'modelspecs', 'rec'], ['modelspecs']])

# GENERATE PLOTS
xfspec.append(['nems.xforms.plot_summary',    {}])

# actually do the fit
ctx, log_xf = xforms.evaluate(xfspec)

"""
