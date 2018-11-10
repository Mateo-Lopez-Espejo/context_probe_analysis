import os
import seaborn as sns
import scipy.stats as sst
import joblib as jl
import numpy as np
import matplotlib.pyplot as plt
import nems
import nems.analysis.api
import nems.initializers
import nems.priors
import nems.recording as recording
import nems.uri
import nems.utils
from nems.fitters.api import scipy_minimize


def split_rec_for_cellwise_fit(rec, stim_sig='stim', resp_sig='resp', pop_in_stim=False, verbose=False):
    '''
    Split a population recording witn n cells into n single cell recordings, adding all othe cells into the
    stimulus signal alongside the sound stimulus ToDo
    :param rec:
    :param stim_sig:
    :param resp_sig:
    :pop_in_stim: Bool, default False. False, discards othe cells from the recordgin. True, uses other cells as predictors
    :return:
    '''

    stim = rec[stim_sig].rasterize()
    resp = rec[resp_sig].rasterize()

    all_cells = rec.meta['cellid']

    recording_list = list()

    if verbose: print('spliting recording wiht {} cells into {} recordings'.format(len(all_cells), len(all_cells)))

    for cc, this_cell in enumerate(all_cells):

        if verbose: print('working in cell {}/{}: {}'.format(cc+1, len(all_cells), this_cell))
        # moves other cell responses from resp to stim. leaves only one cell in resp
        this_cell_data = np.expand_dims(resp._data[cc,:], axis=0)
        others_cells_data = np.delete(resp._data, cc, axis=0)
        if pop_in_stim is True:
            new_stim_data = np.concatenate([stim._data, others_cells_data], axis=0)
        elif pop_in_stim is False:
            new_stim_data = stim._data
        else:
            raise ValueError('pop_in_stim must be bool')

        # updates meta for resp
        new_resp_meta = resp.meta.copy()
        new_resp_meta.update({'cellid':[this_cell]})

        # generates a list of identities for the new channelse of stim
        other_cells_list = [other_cell for other_cell in all_cells if other_cell != this_cell]
        new_stim_chans = ['sound-ch-{}'.format(chan_num) for chan_num in range(stim.nchans)]
        if pop_in_stim is True:
            new_stim_chans.extend(other_cells_list)
        elif pop_in_stim is False:
            pass
        else:
            raise ValueError('pop_in_stim must be bool')

        # updates new singnal fields
        new_stim = stim._modified_copy(data=new_stim_data, chans=new_stim_chans, nchan=len(new_stim_chans))
        new_resp =  resp._modified_copy(data=this_cell_data, meta=new_resp_meta, chans=[this_cell], nchans=1)

        # creates a new recording, with the modified signals, and modified metadata
        new_recording = rec.copy()
        new_recording[stim_sig] = new_stim
        new_recording[resp_sig] = new_resp
        new_recording.meta['cellid'] = [this_cell] # todo, should this be done instead of keeping information of all cells?

        recording_list.append(new_recording)

    return recording_list

def merge_recs_from_cellwise_fit(recs,rec, stim_sig='stim', resp_sig='resp', pop_in_stim=False, verbose=False):
    return None


def test_pop_fit_cellwise():
    # gets test recording, as would be the output of CPP preprocesing
    testfile = '/home/mateo/code/context_probe_analysis/pickles/BRT037b'
    # testfile = 'C:\\Users\\Mateo\\Science\\David_lab\\code\\context_probe_analysis\\pickles\\BRT037b'  # path for blade
    rec = jl.load(testfile)

    # ToDo split into estimation validation sets
    subsets = {'est': rec, 'val': rec}

    # basic modelspec, weighted channels, linear filter and DC shift
    modelspec_name = 'wc.2x2.g-fir.2x15-lvl.1'
    modelspec_name = 'fir.19x15-lvl.1'

    # iterates over each cell, creating a new stim recording containing the response of all other cells in the population
    ntime_modspecs = list()

    for ii, cellid in enumerate(rec.meta['cellid']):

        print('working on cell {}/{}, {}'.format(ii + 1, len(rec.meta['cellid']), cellid))

        working_sets = dict()
        # does the recording construction for each data set
        for key, subset in subsets.items():
            stim = subset['stim'].rasterize()
            resp = subset['resp'].rasterize()

            # splits the response into the cell to be fitted and the cells to use as predictors
            this_cell_resp = np.expand_dims(resp._data[ii, :], 0)
            other_cells_resp = np.delete(resp._data, ii, axis=0)

            # makes the new stmulus array by concatenating stimulus and other cell in the population
            nstim_data = np.concatenate([stim._data, other_cells_resp], axis=0)

            # build signals with modified _data
            stim = stim._modified_copy(data=nstim_data)
            resp = resp._modified_copy(data=this_cell_resp)

            signals = {'stim': stim, 'resp': resp}

            # makes into a recording
            mod_rec = recording.Recording(signals)

            # adds metadata, e
            this_cell_meta = rec.meta.copy()
            this_cell_meta['cellid'] = [cellid]
            mod_rec.meta = this_cell_meta

            # orders in etimation validation sets
            working_sets[key] = mod_rec

        est = working_sets['est']
        val = working_sets['val']

        # parses some data from rec meta into the analysis meta
        analysis_meta = {'modelname': modelspec_name, 'recording': None, **this_cell_meta}
        modelspec = nems.initializers.from_keywords(modelspec_name, meta=analysis_meta)

        # prefits the dc_shift
        modelspec = nems.initializers.prefit_to_target(est, modelspec, nems.analysis.api.fit_basic,
                                                       target_module='levelshift',
                                                       fitter=scipy_minimize,
                                                       fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}})

        # then fit full nonlinear model
        modelspecs = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

        # ----------------------------------------------------------------------------
        # GENERATE SUMMARY STATISTICS

        # generate predictions
        est, val = nems.analysis.api.generate_prediction(est, val, modelspecs)

        # evaluate prediction accuracy
        modelspecs = nems.analysis.api.standard_correlation(est, val, modelspecs)

        ntime_modspecs.append(modelspecs)

        print("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
            modelspecs[0][0]['meta']['r_fit'][0],
            modelspecs[0][0]['meta']['r_test'][0]))

    return ntime_modspecs


def test_cell_fit_cellwise():
    # gets test recording, as would be the output of CPP preprocesing
    testfile = '/home/mateo/code/context_probe_analysis/pickles/BRT037b'
    # testfile = 'C:\\Users\\Mateo\\Science\\David_lab\\code\\context_probe_analysis\\pickles\\BRT037b'  # path for blade
    rec = jl.load(testfile)

    # ToDo split into estimation validation sets
    subsets = {'est': rec, 'val': rec}

    # basic modelspec, weighted channels, linear filter and DC shift
    modelspec_name = 'wc.2x2.g-fir.2x15-lvl.1'
    modelspec_name = 'fir.2x15-lvl.1'

    # iterates over each cell, creating a new stim recording containing the response of all other cells in the population
    ntime_modspecs = list()

    for ii, cellid in enumerate(rec.meta['cellid']):

        print('working on cell {}/{}, {}'.format(ii + 1, len(rec.meta['cellid']), cellid))

        working_sets = dict()
        # does the recording construction for each data set
        for key, subset in subsets.items():
            stim = subset['stim'].rasterize()
            resp = subset['resp'].rasterize()

            # splits the response into the cell to be fitted and the cells to use as predictors
            this_cell_resp = np.expand_dims(resp._data[ii, :], 0)
            other_cells_resp = np.delete(resp._data, ii, axis=0)

            # build signals with modified _data
            resp = resp._modified_copy(data=this_cell_resp)

            signals = {'stim': stim, 'resp': resp}

            # makes into a recording
            mod_rec = recording.Recording(signals)

            # adds metadata, e
            this_cell_meta = rec.meta.copy()
            this_cell_meta['cellid'] = [cellid]
            mod_rec.meta = this_cell_meta

            # orders in etimation validation sets
            working_sets[key] = mod_rec

        est = working_sets['est']
        val = working_sets['val']

        # parses some data from rec meta into the analysis meta
        analysis_meta = {'modelname': modelspec_name, 'recording': None, **this_cell_meta}
        modelspec = nems.initializers.from_keywords(modelspec_name, meta=analysis_meta)

        # prefits the dc_shift
        modelspec = nems.initializers.prefit_to_target(est, modelspec, nems.analysis.api.fit_basic,
                                                       target_module='levelshift',
                                                       fitter=scipy_minimize,
                                                       fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}})

        # then fit full nonlinear model
        modelspecs = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

        # ----------------------------------------------------------------------------
        # GENERATE SUMMARY STATISTICS

        # generate predictions
        est, val = nems.analysis.api.generate_prediction(est, val, modelspecs)

        # evaluate prediction accuracy
        modelspecs = nems.analysis.api.standard_correlation(est, val, modelspecs)

        ntime_modspecs.append(modelspecs)

        print("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
            modelspecs[0][0]['meta']['r_fit'][0],
            modelspecs[0][0]['meta']['r_test'][0]))

    return ntime_modspecs


def test_cell_fit_all():
    # gets test recording, as would be the output of CPP preprocesing
    testfile = '/home/mateo/code/context_probe_analysis/pickles/BRT037b'
    # testfile = 'C:\\Users\\Mateo\\Science\\David_lab\\code\\context_probe_analysis\\pickles\\BRT037b'  # path for blade
    rec = jl.load(testfile)

    # ToDo split into estimation validation sets
    subsets = {'est': rec, 'val': rec}

    # basic modelspec, weighted channels, linear filter and DC shift
    modelspec_name = 'wc.2x2.g-fir.2x15-lvl.1'
    modelspec_name = 'fir.2x15-lvl.18'

    est = rec
    val = rec

    # parses some data from rec meta into the analysis meta
    analysis_meta = {'modelname': modelspec_name, 'recording': rec, **rec.meta}
    modelspec = nems.initializers.from_keywords(modelspec_name, meta=analysis_meta)

    # prefits the dc_shift
    modelspec = nems.initializers.prefit_to_target(est, modelspec, nems.analysis.api.fit_basic,
                                                   target_module='levelshift',
                                                   fitter=scipy_minimize,
                                                   fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}})

    # then fit full nonlinear model
    modelspecs = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize)

    # ----------------------------------------------------------------------------
    # GENERATE SUMMARY STATISTICS

    # generate predictions
    est, val = nems.analysis.api.generate_prediction(est, val, modelspecs)

    # evaluate prediction accuracy
    modelspecs = nems.analysis.api.standard_correlation(est, val, modelspecs)

    print("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
        modelspecs[0][0]['meta']['r_fit'][0],
        modelspecs[0][0]['meta']['r_test'][0]))

    return modelspecs


# dirty cache
# todo, create function create a name and cache if name does not exist
def make_chache(func, filename):

    if os.path.exists(filename):
        print('cache exists, returning path')
        toload = filename

    else:
        print('no cache, calculating')
        ntime_modspecs = func()
        print('making cache, and returning path')
        jl.dump(ntime_modspecs, filename)
        toload = filename
    return toload


def get_cache(toload):
    return jl.load(toload)

###############################################
# fits the models and hold on cache

# performs the cellwise population fit, cell by cell
pop_fit_cachename = '/home/mateo/code/context_probe_analysis/pickles/pop_fit_modspecs'
pop_fit_file = make_chache(test_pop_fit_cellwise, pop_fit_cachename)
pop_modspecs = get_cache(pop_fit_file)

# performs the single cell, population independent fit, cell by cell
cell_fit_cachename = '/home/mateo/code/context_probe_analysis/pickles/cells_fit_modspecs'
cell_fit_file = make_chache(test_cell_fit_cellwise, cell_fit_cachename)
cells_modspecs = get_cache(cell_fit_file)

# performs te single cell, population independent fit, on all cells simultaneously
all_fit_cachename = '/home/mateo/code/context_probe_analysis/pickles/all_fit_modspecs'
all_fit_file = make_chache(test_cell_fit_all, all_fit_cachename)
all_modspecs = get_cache(all_fit_file)

###############################################
# extract fited values in a workable format
# population
pop_vals = {'fir': [], 'lvl': []}
for mm, this_mod in enumerate(pop_modspecs):
    pop_vals['fir'].append(this_mod[0][0]['phi']['coefficients'])
    pop_vals['lvl'].append(this_mod[0][1]['phi']['level'])

# single cell, cell by cell
cell_vals = {'fir': [], 'lvl': []}
for mm, this_mod in enumerate(cells_modspecs):
    cell_vals['fir'].append(this_mod[0][0]['phi']['coefficients'])
    cell_vals['lvl'].append(this_mod[0][1]['phi']['level'])

# single cell, simultaneously,
all_vals = {}
all_vals['fir'] = all_modspecs[0][0]['phi']['coefficients']
all_vals['lvl'] = all_modspecs[0][1]['phi']['level']
# it seems that when fitting all cells together, a signle FIR, one size fits all is used. This precludes further analisi
# on individual cells ToDo ask stephen what is the logic behind fitting a single FIR.


# compares the common paramter values between the population and cell fittings i.e. sound FIR and lvl
linearized_pop_fir  = list()
linearized_cell_fir = list()
for cell in range(len(pop_vals['fir'])):
    linearized_pop_fir.extend(pop_vals['fir'][cell][0:2,:].flatten().tolist())
    linearized_cell_fir.extend(cell_vals['fir'][cell].flatten().tolist())

linearized_pop_lvl  = list()
linearized_cell_lvl = list()
for cell in range(len(pop_vals['fir'])):
    linearized_pop_lvl.extend(pop_vals['lvl'][cell].flatten().tolist())
    linearized_cell_lvl.extend(cell_vals['lvl'][cell].flatten().tolist())

linearized_pop_fir  = np.asarray(linearized_pop_fir)
linearized_cell_fir = np.asarray(linearized_cell_fir)
linearized_pop_lvl  = np.asarray(linearized_pop_lvl)
linearized_cell_lvl = np.asarray(linearized_cell_lvl)

# plots the differences

fig, axes = plt.subplots(1, 2)
axes = np.ravel(axes)

sns.regplot(linearized_pop_fir, linearized_cell_fir, ax=axes[0])
fir_reg = sst.linregress(linearized_pop_fir, linearized_cell_fir)
print('fir reg: {}'.format(fir_reg))
sns.regplot(linearized_pop_lvl, linearized_cell_lvl, ax=axes[1])
lvl_reg = sst.linregress(linearized_pop_lvl, linearized_cell_lvl)
print('lvl reg: {}'.format(lvl_reg))

axes[0].set_title('FIR values')
axes[1].set_title('lvl values')


for ax in axes:
    ax.set_xlabel('population calculation')
    ax.set_ylabel('cell wise calculation')
    ax.set_ylim(ax.get_xlim())
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')

###############################################
# compare the goodness of fit
# parses the goodness of fit into a workable data structure
pop_rtest = np.asarray([cell[0][0]['meta']['r_test'][0] for cell in pop_modspecs])
cell_rtest = np.asarray([cell[0][0]['meta']['r_test'][0] for cell in cells_modspecs])
all_rtest = all_modspecs[0][0]['meta']['r_test']

# makes scatterplots
fig, axes = plt.subplots(1,3)
axes = np.ravel(axes)

sns.regplot(all_rtest, cell_rtest, ax=axes[0])

sns.regplot(all_rtest, pop_rtest, ax=axes[1])

sns.regplot(cell_rtest, pop_rtest, ax=axes[2])

axes[0].set_xlabel('cells simultaneous fit')
axes[0].set_ylabel('cells cellwise fit')

axes[1].set_xlabel('cells simultaneous fit')
axes[1].set_ylabel('population fit')

axes[2].set_xlabel('cells cellwise fit')
axes[2].set_ylabel('population fit')

for ax in axes:
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')


fig.suptitle('r_test comparison between different modelse/fitting')

###############################################
# predicts response based on params. calculates the contextual effect for actual vs predicted
# first generates the split recording
rec = jl.load('/home/mateo/code/context_probe_analysis/pickles/BRT037b')

cell_recs = split_rec_for_cellwise_fit(rec, pop_in_stim=False)
pop_recs = split_rec_for_cellwise_fit(rec, pop_in_stim=True)

# then pairs the correc recording with modelspec, and calculates the prediction
cell_preds = list ()
for c_rec, c_modspec in zip(cell_recs, cells_modspecs):
    c_pred = nems.analysis.api.generate_prediction(c_rec, c_rec, c_modspec)
    cell_preds.append(c_pred[0][0])

pop_preds = list ()
for p_rec, p_modspec in zip(pop_recs, pop_modspecs):
    p_pred = nems.analysis.api.generate_prediction(p_rec, p_rec, p_modspec)
    pop_preds.append(p_pred[0][0])
