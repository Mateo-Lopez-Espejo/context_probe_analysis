from nems_db.xform_wrappers import baphy_load_wrapper
import sqlalchemy
import collections as coll
import pandas as pd
import nems.db as nd
from nems.recording import load_recording
import nems.preprocessing as preproc
import nems.epoch as ne
import numpy as np
import matplotlib.pyplot as plt

'''
This script explore how active are all the cells recordes with natural sounds, to specific natural sounds in the library.
It serves to select a set of the most driving sounds to generate the new soundobject CPN (context probe natural), which
is the second iteration of CPP
'''



# cellid="AMT003c-11-1"
batch=289
loadkey="ozgf.fs100.ch18"



# I need to figure out a better way of  getting this values than copy them from matlab
# using   cfd=dbgetscellfile('runclass','NAT','Ref_Subsets','=3');
# This is temporary

cell_list = ['AMT003c-11-1', 'AMT003c-14-1', 'AMT003c-19-1', 'AMT003c-20-1', 'AMT003c-25-1', 'AMT003c-26-1',
             'AMT003c-32-1', 'AMT003c-32-2', 'AMT003c-33-1', 'AMT003c-37-1', 'AMT003c-42-1', 'AMT003c-46-1',
             'AMT003c-49-1', 'AMT005c-02-1', 'AMT005c-02-2', 'AMT005c-04-1', 'AMT005c-08-1', 'AMT005c-11-1',
             'AMT005c-12-1', 'AMT005c-13-1', 'AMT005c-13-2', 'AMT005c-14-1', 'AMT005c-19-1', 'AMT005c-20-1',
             'bbl099g-04-1', 'bbl099g-05-1', 'bbl099g-06-1', 'bbl099g-08-1', 'bbl099g-09-1', 'bbl099g-09-2',
             'bbl099g-20-1', 'bbl099g-20-2', 'bbl099g-25-1', 'bbl099g-28-1', 'bbl099g-30-1', 'bbl099g-30-2',
             'bbl099g-31-1', 'bbl099g-33-1', 'bbl099g-40-1', 'bbl099g-46-1', 'bbl099g-49-1', 'bbl099g-51-1',
             'bbl099g-52-1', 'bbl099g-54-1', 'bbl099g-55-1', 'bbl099g-57-1', 'bbl099g-58-1', 'bbl104h-02-1',
             'bbl104h-05-1', 'bbl104h-06-1', 'bbl104h-06-2', 'bbl104h-07-1', 'bbl104h-08-1', 'bbl104h-10-1',
             'bbl104h-10-2', 'bbl104h-12-1', 'bbl104h-13-1', 'bbl104h-16-1', 'bbl104h-16-2', 'bbl104h-19-1',
             'bbl104h-20-1', 'bbl104h-22-1', 'bbl104h-28-1', 'bbl104h-30-1', 'bbl104h-33-1', 'bbl104h-34-1',
             'bbl104h-37-1', 'bbl104h-42-1', 'bbl104h-44-1', 'bbl104h-47-1', 'bbl104h-48-1', 'bbl104h-50-1',
             'bbl104h-64-1', 'bbl106d-04-1', 'bbl106d-16-1', 'bbl106d-29-1', 'bbl106d-35-1', 'bbl106d-40-1',
             'bbl106d-46-1', 'bbl106d-49-1', 'bbl106d-52-1', 'bbl106d-58-1', 'BRT026c-01-1', 'BRT026c-02-4',
             'BRT026c-02-5', 'BRT026c-05-3', 'BRT026c-08-2', 'BRT026c-16-2', 'BRT026c-17-3', 'BRT026c-19-2',
             'BRT026c-22-1', 'BRT026c-23-1', 'BRT026c-25-3', 'BRT026c-27-2', 'BRT026c-29-2', 'BRT026c-30-1',
             'BRT026c-31-2', 'BRT026c-33-2', 'BRT026c-34-2', 'BRT026c-41-2', 'BRT026c-44-2', 'BRT026c-46-1',
             'BRT026c-46-2', 'BRT026c-47-1', 'BRT032e-15-1', 'BRT032e-18-1', 'BRT032e-24-1', 'BRT032e-25-1',
             'BRT032e-30-1', 'BRT032e-31-1', 'BRT032e-33-1', 'BRT032e-37-1', 'BRT032e-37-2', 'BRT032e-39-1',
             'BRT033b-01-1', 'BRT033b-03-2', 'BRT033b-03-3', 'BRT033b-04-1', 'BRT033b-09-2', 'BRT033b-12-3',
             'BRT033b-12-4', 'BRT033b-15-2', 'BRT033b-16-1', 'BRT033b-17-1', 'BRT033b-29-1', 'BRT033b-32-1',
             'BRT033b-35-1', 'BRT033b-36-1', 'BRT033b-44-1', 'BRT033b-50-3', 'BRT033b-51-1', 'BRT033b-53-1',
             'BRT033b-58-1', 'BRT034f-01-1', 'BRT034f-02-2', 'BRT034f-07-2', 'BRT034f-34-2', 'BRT034f-36-1',
             'BRT034f-38-1', 'BRT034f-39-2', 'BRT034f-42-1', 'BRT034f-45-1', 'BRT034f-54-1', 'BRT034f-54-2',
             'BRT034f-54-3', 'BRT034f-57-1', 'BRT036b-03-1', 'BRT036b-06-2', 'BRT036b-16-1', 'BRT036b-21-2',
             'BRT036b-28-2', 'BRT036b-31-2', 'BRT036b-37-1', 'BRT036b-39-1', 'BRT036b-45-2', 'BRT036b-49-1',
             'BRT036b-49-2', 'BRT037b-13-1', 'BRT037b-24-2', 'BRT037b-28-1', 'BRT037b-30-1', 'BRT037b-33-1',
             'BRT037b-36-1', 'BRT037b-36-2', 'BRT037b-43-1', 'BRT037b-63-1', 'BRT038b-05-1', 'BRT038b-08-1',
             'BRT038b-11-1', 'BRT038b-11-2', 'BRT038b-12-1', 'BRT038b-16-1', 'BRT038b-17-1', 'BRT038b-20-1',
             'BRT038b-20-2', 'BRT038b-22-1', 'BRT038b-23-1', 'BRT038b-26-1', 'BRT038b-30-1', 'BRT038b-39-1',
             'BRT038b-42-1', 'BRT038b-43-1', 'BRT038b-45-1', 'BRT038b-55-1', 'BRT039c-05-1', 'BRT039c-06-1',
             'BRT039c-14-1', 'BRT039c-21-2', 'BRT039c-23-1', 'BRT039c-23-2', 'BRT039c-25-2', 'BRT039c-25-3',
             'BRT039c-26-1', 'BRT039c-26-2', 'BRT039c-33-2', 'BRT039c-33-3', 'BRT039c-39-1', 'BRT039c-39-2',
             'BRT039c-48-2', 'BRT039c-50-1', 'BRT039c-52-1']

# gets the first cell for eache site since I am simulatenously loading all cells for a given site
site_dict = coll.defaultdict(list)
for cell in cell_list:
    cell_site = cell.split('-')[0]
    site_dict[cell_site].append(cell)

first_cell = [cells[0] for cells in site_dict.values()]


########################################################################################################################
# for each stimulus defines a histogram showing the number of spikes generated for each cell

# crawls each site, crawls each neuron in a site,
# calculates a norm_metric of activity, in this case takes the PSTH and integrates the firing over time
# saves the value in an DataFrame with dimensions Stimulus Cell
# for ease in calculations, saves the valuese in a 3d array with dimentions Sound x Cellid x Time.
# given that time is different across different cells finds the max dimentions of the array, initializes with nan
# and orderly fills with values

site_DFs = list()
bad_cells = list()
mats_dict = coll.defaultdict(lambda: coll.defaultdict(list))
cells_sets = dict()
for cellid in first_cell:

    # cellid='bbl099g-04-1'
    try:
        ctx=baphy_load_wrapper(cellid=cellid, batch=batch, loadkey=loadkey, siteid=None)
        rec=load_recording(ctx['recording_uri_list'][0])
    except:
        print('could not load site for cell {}'.format(cellid))
        bad_cells.append(cellid)
        continue

    sig = rec['resp'].rasterize()

    #wav_names = ne.epoch_names_matching(sig.epochs, r'^STIM_00') # for validation set only
    wav_names = ne.epoch_names_matching(sig.epochs, r'^STIM_')
    epre = sig.get_epoch_indices('PreStimSilence')
    epost = sig.get_epoch_indices('PostStimSilence')
    prebins = epre[0][1] - epre[0][0]
    postbins = epost[0][1] - epost[0][0]

    mtx_dict = sig.extract_epochs(wav_names)
    # orderse the response of all the cells in the site to all the sounds in a 3d array of shape Sound x Cells x Time
    # then normalizes the response of each cell across all stimuli
    shape = [len(mtx_dict.keys()),
             list(mtx_dict.values())[0].shape[1],
             np.max([val.shape[2] for val in mtx_dict.values()])]
    site_arr = np.empty(shape); site_arr[:] = np.nan
    for ss, (sound, response) in enumerate(mtx_dict.items()):
        PSTH = np.mean(response, axis=0)
        site_arr[ss, :, :] = PSTH

    site_arr = site_arr[:,:,prebins:-postbins]
    #site_arr = site_arr[:,:,prebins:(prebins+100)]

    # normalizese the responses time and stimulus
    mean = np.nanmean(site_arr, axis=(0,2), keepdims=True)
    std = np.nanstd(np.nanmean(site_arr, axis=2, keepdims=True), axis=0, keepdims=True)
    norm_arr = (site_arr - mean) / std
    #norm_arr = site_arr
    norm_metric = np.nanmean(site_arr, axis=2)
    norm_metric -= np.mean(norm_metric, axis=0, keepdims=True)
    norm_metric /= np.std(norm_metric, axis=0, keepdims=True)

    # sets in a site mini df, to keep track of cellid and soundid
    df = pd.DataFrame(data=norm_metric, index=mtx_dict.keys(), columns=sig.chans)
    site_DFs.append(df)
    cells_sets[cellid] = sig.chans

    # saves the PSTHs of cells to a growing dictionarie ordered by sound id, for later plot all cells for a sound.
    for si, sound in enumerate(mtx_dict.keys()):
        mats_dict[sound]['mtx'].append(norm_arr[si,:,:])
        mats_dict[sound]['cellid'].extend(sig.chans)

raw_DF = pd.concat(site_DFs, axis=1, join='inner')
cellids = raw_DF.columns.tolist()
soundnames = raw_DF.index.tolist()

# for a given stimulus, concatenates the responses of all cell, across all site. incongruent response lengths are padded with nan
# first figure out the number of cells and the maximum time length to initialize an empty array
for outkey, indict in mats_dict.items():
    cellcount = 0
    time_lengths = list()
    for site in indict['mtx']:
        cellcount += site.shape[0]
        time_lengths.append(site.shape[1])

    all_cells_mtx = np.empty([cellcount, np.max(time_lengths)])
    all_cells_mtx[:] = np.nan

    offset = 0
    for ss, site in enumerate(indict['mtx']):
        # normalizes the site

        all_cells_mtx[offset:offset+site.shape[0], 0:site.shape[1]] = site
        offset = offset + site.shape[0]

    mats_dict[outkey]['full_mtx'] = all_cells_mtx


# calculates the mean response for eache stimulus (across all cells),
DF = raw_DF.copy()
DF['mean'] = raw_DF.apply(np.mean, axis=1)
DF['std'] = raw_DF.apply(np.std, axis=1)
sorted = DF.sort_values(by=['mean'], axis=0, ascending=False)
filtered = sorted.loc[:, cellids]

fig, axes = plt.subplots(1, 3)
axes = np.ravel(axes)
axes[0].hist(DF['mean'].values, bins=50)
axes[0].set_title('mean')
axes[1].hist(DF['std'].values, bins=50)
axes[1].set_title('std')
axes[2].imshow(filtered.values, aspect='auto')
axes[2].set_ylabel('sounds')
axes[2].set_xlabel('cells')
axes[2].set_title('full dataset, summed activity, sorted by sound')

fig.suptitle('all stimuli distribution (collapsed across cells)')


# selects the n sounds with the highest value
# plots an histograms of the distributions of individual cell responses, alongside the spectrogram of the sound,
# and raster plot of all the cells mean response over time, sorted by integrated response

spectrograms = rec['stim']._data

DF = DF.sort_values(by=['mean'], axis=0, ascending=False)
subset = DF.tail(4)

for sound, row in subset.iterrows():
    fig= plt.figure()
    ax0 = plt.subplot2grid((2, 2), (0, 0))
    ax1 = plt.subplot2grid((2, 2), (0, 1))
    ax2 = plt.subplot2grid((2, 2), (1,0), colspan=2)
    ax0.imshow(spectrograms[row.name], aspect='auto')
    ax0.set_title('stimulus spectrogram')
    ax1.hist(row.values, bins=50)
    ax1.set_title('cellwise summed-response histogram')

    # sorts the matrix for the raster plot
    full_mtx = mats_dict[row.name]['full_mtx']

    integrated =  np.nansum(full_mtx, axis=1)
    idx_sort = np.argsort(integrated)
    sort_mtx = full_mtx[idx_sort]

    ax2.imshow(sort_mtx, aspect='auto')
    ax2.set_title('sorted cellwise raster')
    fig.suptitle(sound)

selected_sounds = subset.index.tolist()


#######
# this is just copy pasting the result of the previous operation, for the sake of easy access
Most_driving_sounds = ['STIM_cat599_rec1_factory_sounds_excerpt1.wav',
                       'STIM_cat220_rec1_latin-pop_enrique-iglesias_i-will-survive_excerpt1.wav',
                       'STIM_cat592_rec1_pouring_cereal_excerpt1.wav',
                       'STIM_cat437_rec1_pop_kelly-clarkson_einstein_excerpt1.wav',
                       'STIM_cat173_rec1_giggling_excerpt1.wav',
                       'STIM_cat409_rec1_window_blinds_excerpt1.wav',
                       'STIM_cat66_rec1_cash_register_excerpt1.wav',
                       'STIM_cat90_rec1_colouring_freesound_123jorre456_excerpt1.wav',
                       'STIM_cat668_rec2_ferret_fights_Jasmine-Violet001_excerpt2.wav',
                       'STIM_cat183_rec1_grunt_groan_excerpt1.wav',
                       'STIM_cat328_rec1_shaving_excerpt1.wav',
                       'STIM_cat159_rec1_flute_bourree_excerpt1.wav',
                       'STIM_cat668_rec1_ferret_fights_Athena-Violet001_excerpt2.wav',
                       'STIM_cat148_rec1_fiddle_luke-abbott-willie_moore_excerpt1.wav',
                       'STIM_cat361_rec1_tambourine_excerpt1.wav',
                       'STIM_cat370_rec1_toothbrushing_excerpt1.wav']
