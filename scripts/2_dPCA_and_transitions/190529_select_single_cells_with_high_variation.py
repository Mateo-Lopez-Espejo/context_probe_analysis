import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import epochs as cpe, triplets as tp
import nems.recording as recording
import nems_lbhb.baphy as nb

'''
rough selection of good cells based on arbitrary threshold on their response amplitude (mean)
and response variation (std) dependent on context. for each probe for each site 
'''

site = 'AMT032a'  # great site. PEG

modelname = 'resp'
options = {'batch': 316,
           'siteid': site,
           'stimfmt': 'envelope',
           'rasterfs': 100,
           'recache': False,
           'runclass': 'CPN',
           'stim': False}  # ToDo chace stims, spectrograms???

load_URI = nb.baphy_load_recording_uri(**options)
loaded_rec = recording.load_recording(load_URI)

rec = cpe.set_recording_subepochs(loaded_rec, set_pairs=True)
sig = rec['resp']
eps = sig.epochs

# AMT032a
goodcells = ['AMT032a-12-1', 'AMT032a-15-1', 'AMT032a-17-1', 'AMT032a-21-1', 'AMT032a-26-2', 'AMT032a-28-1',
             'AMT032a-34-2', 'AMT032a-38-1', 'AMT032a-38-2', 'AMT032a-40-1', 'AMT032a-40-2', 'AMT032a-41-1',
             'AMT032a-44-1']
best_cell = 'AMT032a-40-2'

# results_file = nd.get_results_file(316)  # cpp batch.

all_sites = ['ley070a', 'ley072b', 'AMT028b', 'AMT029a', 'AMT030a', 'AMT031a', 'AMT032a']

df = list()


for site in all_sites:
    modelname = 'resp'
    options = {'batch': 316,
               'siteid': site,
               'stimfmt': 'envelope',
               'rasterfs': 100,
               'recache': False,
               'runclass': 'CPN',
               'stim': False}  # ToDo chace stims, spectrograms???

    try:
        load_URI = nb.baphy_load_recording_uri(**options)
        loaded_rec = recording.load_recording(load_URI)
    except:
        print(f'failed importing{site}')
        continue

    rec = cpe.set_recording_subepochs(loaded_rec, set_pairs=True)
    sig = rec['resp'].rasterize()
    eps = sig.epochs

    # define wich cells present more context induced variability
    # Context x Probe x Repetition x Unit x Time
    full_array, bad_cpp, good_cpp, context_names, probe_names = tp.make_full_array(sig, 'CPN')
    full_PSTH = np.nanmean(full_array, axis=2)  # collapses across repetitions
    probe_PSTH = full_PSTH[:, :, :, int(np.floor(full_PSTH.shape[-1] / 2)):]  # takes probe response i.e second half
    # calculates probe wise values
    # calculates cell single number context driven probe response variation
    ctx_vars = np.nanstd(probe_PSTH, axis=0)  # collapses across contexts
    probes_var_over_time = np.nanmean(ctx_vars, axis=-1)  # collapses across time

    # calculate cell single number probe response amplitude
    ctx_amps = np.nanmean(probe_PSTH, axis=0)  # collapses across contexts
    probes_amp_over_time = np.nanmean(ctx_amps, axis=-1)  # collapses across time
    for cc, cellid in enumerate(sig.chans):

        for pp, probe in enumerate(probe_names):
            d = {'cellid': cellid,
                 'epoch': probe,
                 'parameter': 'std',
                 'value': probes_var_over_time[pp, cc]}
            df.append(d)

            d = {'cellid': cellid,
                 'epoch': probe,
                 'parameter': 'mean',
                 'value': probes_amp_over_time[pp, cc]}
            df.append(d)

    ##########################################################################################
    # Uses silence as baseline activity for mean and standard deviation of z-score calculation
    silences = sig.extract_epochs(['PreStimSilence', 'PostStimSilence'])
    silences = np.concatenate(list(silences.values()), axis=0)  # shape Reps x Cells x Time

    silence_mean = np.nanmean(silences, axis=(0, 2))
    silence_std = np.nanstd(silences, axis=(0, 2))

    # stores silence meand and std to df
    for cc, cellid in enumerate(sig.chans):
        d = {'cellid': cellid,
             'epoch': 'silence',
             'parameter': 'mean',
             'value': silence_mean[cc]}
        df.append(d)

        d = {'cellid': cellid,
             'epoch': 'silence',
             'parameter': 'std',
             'value': silence_std[cc]}
        df.append(d)

    full_zscores = np.empty(full_array.shape)
    full_zscores[:] = np.nan
    for cc in range(full_zscores.shape[3]):  # iterates over the cell dimension
        full_zscores[:, :, :, cc, :] = (full_array[:, :, :, cc, :] - silence_mean[cc]) / silence_std[cc]

    # calculates mean and std for the zscore as done before for the raw values

    zscore_PSTH = np.nanmean(full_zscores, axis=2)  # collapses across repetitions
    z_probe_PSTH = zscore_PSTH[:, :, :,
                               int(np.floor(zscore_PSTH.shape[-1] / 2)):]  # takes probe response i.e second half

    # calculates probe wise values
    # calculates cell single number context driven probe response variation
    z_ctx_vars = np.nanstd(z_probe_PSTH, axis=0)  # collapses across contexts
    z_probes_var_over_time = np.nanmean(z_ctx_vars, axis=-1)  # collapses across time

    # calculate cell single number probe response amplitude
    z_ctx_amps = np.nanmean(z_probe_PSTH, axis=0)  # collapses across contexts
    z_probes_amp_over_time = np.nanmean(z_ctx_amps, axis=-1)  # collapses across time
    for cc, cellid in enumerate(sig.chans):

        for pp, probe in enumerate(probe_names):
            d = {'cellid': cellid,
                 'epoch': probe,
                 'parameter': 'zstd',
                 'value': z_probes_var_over_time[pp, cc]}
            df.append(d)

            d = {'cellid': cellid,
                 'epoch': probe,
                 'parameter': 'zscore',
                 'value': z_probes_amp_over_time[pp, cc]}
            df.append(d)

DF = pd.DataFrame(df)
wdf = DF.copy()
wdf['site'] = wdf['cellid'].str.slice(0, 7, 1)

sites = wdf.site.unique()

########################################################################################################################
# plots amplitude response vs context induced variance for individual probes, and the mean across probes, for each site
fig, axes = plt.subplots()
x_ax = 'zscore'
y_ax = 'zstd'
x_thr = 0.18
y_thr = 0

selected_cells = list()

for ss, site in enumerate(sites):
    ffsite = wdf.site == site
    ffamp = wdf.parameter == x_ax
    ffvar = wdf.parameter == y_ax
    ffepoch = wdf.epoch.str.match(r'\AP\d*')

    amps = wdf.loc[ffsite & ffamp & ffepoch, :]
    vars = wdf.loc[ffsite & ffvar & ffepoch, :]

    p_amps = amps.pivot(index='epoch', columns='cellid', values='value')
    p_vars = vars.pivot(index='epoch', columns='cellid', values='value')

    X = np.mean(p_amps, axis=0)
    Y = np.mean(p_vars, axis=0)

    sss = X.index.values[(X >= x_thr) & (Y >= y_thr)]
    selected_cells.extend(sss)

    # defines error bars a the range of the data i.e. min and max values
    Xerr = np.stack([np.mean(p_amps.values, axis=0) - np.min(p_amps.values, axis=0),
                     np.max(p_amps.values, axis=0) - np.mean(p_amps.values, axis=0)], axis=0)
    Yerr = np.stack([np.mean(p_vars.values, axis=0) - np.min(p_vars.values, axis=0),
                     np.max(p_vars.values, axis=0) - np.mean(p_vars.values, axis=0)], axis=0)

    # Xerr = np.std(p_amps.values, axis=0)
    # Yerr = np.std(p_vars.values, axis=0)

    axes.errorbar(X, Y, xerr=Xerr, yerr=Yerr, fmt='o', color=f'C{ss}', ecolor=f'C{ss}', label=site)
    axes.axvline(x_thr, linestyle='--', color='black')
    axes.axhline(y_thr, linestyle='--', color='black')
    axes.legend()
    axes.set_xlabel(x_ax)
    axes.set_ylabel(y_ax)
    axes.set_title('probe response and context driven variation per cell')

print(len(selected_cells))
print(selected_cells)

########################################################################################################################
# plots individual cells (color) probe responses, and mean across probes (transparent, bold) axes = np.ravel(axes)
# for all different sites (subplots)
fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
axes = np.ravel(axes)
x_ax = 'mean'
y_ax = 'std'
for ss, site in enumerate(sites):

    site_DF = wdf.loc[wdf.site == site, :]
    site_cells = site_DF.cellid.unique()

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for (cc, cellid), color in zip(enumerate(site_cells), colors):
        # if cc == 9:
        #     continue

        ffsite = wdf.site == site
        ffcell = wdf.cellid == cellid
        ffepoch = wdf.epoch.str.match(r'\AP\d*')
        ffparam = wdf.parameter.isin([x_ax, y_ax])

        filtered = wdf.loc[ffsite & ffcell & ffepoch & ffparam, :]
        pivoted = filtered.pivot(index='epoch', columns='parameter', values='value')
        mean_probes = np.nanmean(pivoted, axis=0)
        x_mean = np.nanmean(pivoted[x_ax])
        y_mean = np.nanmean(pivoted[y_ax])
        axes[ss].scatter(pivoted[x_ax].values, pivoted[y_ax].values, color=color, alpha=0.3)
        axes[ss].scatter(x_mean, y_mean, color=color, alpha=1)
    axes[ss].set_title(site)
    # axes[ss].legend()

########################################################################################################################
