import nems.xforms as xforms
import nems.modelspec as ms

def single_cpp_processing(cellid, batch, modelspec_name):
    loadkey = "env.fs100.cst"

    options = {'stimfmt': 'envelope',
               'rasterfs': 100,
               'recache': False,
               'runclass': 'CPP',
               'stim': True}

    meta = {'cellid': cellid, 'batch': batch,
            'modelname': modelspec_name, 'recording': cellid}

    xfspec = list()
    xfspec.append(['nems_db.xform_wrappers.baphy_load_wrapper', {'cellid':cellid, 'batch':batch, 'loadkey':loadkey, 'options':options}])
    xfspec.append(['nems.xforms.load_recordings', {'cellid':cellid, 'save_other_cells_to_state':True}])
    xfspec.append(['nems.xforms.split_by_occurrence_counts', {'epoch_regex':'^STIM_'}])
    # xfspec.append(['nems.xforms.average_away_stim_occurrences', {}])
    xfspec.append(['nems.xforms.init_from_keywords', {'keywordstring':modelspec_name, 'meta':meta}])
    xfspec.append(['nems.xforms.fit_basic_init', {}])
    xfspec.append(['nems.xforms.fit_basic', {}])
    xfspec.append(['nems.xforms.predict', {}])
    xfspec.append(['nems.xforms.add_summary_statistics', {}])
    xfspec.append(['nems.xforms.plot_summary', {}])

    ctx, log_xf = xforms.evaluate(xfspec)
    modelspecs = ctx['modelspecs']
    destination = '/auto/data/nems_db/results/{0}/{1}/{2}/'.format(
        batch, cellid, ms.get_modelspec_longname(modelspecs[0]))
    modelspecs[0][0]['meta']['modelpath'] = destination
    modelspecs[0][0]['meta']['figurefile'] = destination + 'figure.0000.png'
    modelspecs[0][0]['meta'].update(meta)

    save_data = xforms.save_analysis(destination,
                             recording=ctx['rec'],
                             modelspecs=modelspecs,
                             xfspec=xfspec,
                             figures=ctx['figures'],
                             log=log_xf)

    print('saving at {}'.format(save_data['savepath']))
    return save_data['savepath']
