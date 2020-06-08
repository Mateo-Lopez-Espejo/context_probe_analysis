import numpy as np
import scipy.stats as sst
import pandas as pd
import joblib as jl
import pathlib as pl
import matplotlib.pyplot as plt
import fancy_plots as fplt
import seaborn as sns
from configparser import ConfigParser
from cpp_cache import set_name

config = ConfigParser()
if pl.Path('../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../context_probe_analysis/config/settings.ini'))
elif pl.Path('../../../context_probe_analysis/config/settings.ini').exists():
    config.read(pl.Path('../../../context_probe_analysis/config/settings.ini'))
else:
    raise FileNotFoundError('config file coluld not be foud')

meta = {'reliability': 0.1,  # r value
        'smoothing_window': 0,  # ms
        'raster_fs': 30,
        'transitions': ['silence', 'continuous', 'similar', 'sharp'],
        'montecarlo': 1000,
        'zscore': True,
        'dprime_absolute':None}

# transferable plotting parameters
plt.rcParams['svg.fonttype'] = 'none'
sup_title_size = 30
sub_title_size = 20
ax_lab_size = 15
ax_val_size = 11
full_screen = [19.2, 9.83]
sns.set_style("ticks")

########################################################################################################################
########################################################################################################################
# data frame containing all the important summary data, i.e. exponential decay fits for dprime and significance, for
# all combinantions of transition parirs, and probes,  for the means across probes, transistion pairs or for both, and
# for the single cell analysis or the dPCA projections

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / 'DF_summary' /set_name(meta)
print('loading cached summary DataFrame')
DF = jl.load(summary_DF_file)

########################################################################################################################
# SC mean vs dPCA taus, tau outliers filtered

ff_anal = DF.analysis == 'single_cell'
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
ff_outliers = DF.value < 2000
sing = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
              ['region', 'siteid', 'cellid', 'parameter', 'value']]

sing_pivot = sing.pivot(index='siteid', columns='cellid', values='value')

sing_pivot['max'] = sing_pivot.mean(axis=1)


ff_anal = DF.analysis == 'dPCA'
pops = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source & ff_outliers,
              ['region', 'siteid', 'cellid', 'parameter', 'value']]

pops = pops.set_index('siteid')

toplot = pd.concat((pops.loc[:, ['region', 'value']], sing_pivot.loc[:, 'max']), axis=1)

fig, ax = plt.subplots()
ax = sns.regplot(x ='value', y= 'max', data=toplot, color='black', ax=ax)
sns.despine(ax=ax)
_,_,r2,_,_ = sst.linregress(toplot['value'], toplot['max'])
_ = fplt.unit_line(ax, square_shape=False)

ax.set_xlabel(f'dPCA tau (ms)', fontsize=ax_lab_size)
ax.set_ylabel(f'single cell mean tau (ms)', fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6,6))
title = 'SC mean dPCA tau comparison r={:.2f}'.format(r2)
fig.suptitle(title, fontsize=sub_title_size)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'DAC3_figures', title)

########################################################################################################################
# creates data frame with rows == cellid and columns == single cell dprime-r0, significance-tau, and their population
# equivalents
ff_probe = DF.probe == 'mean'
ff_trans = DF.transition_pair == 'mean'

# single cell array
ff_anal = DF.analysis == 'single_cell'
# tau
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
SC_tau = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                  ['region', 'siteid', 'cellid', 'value']]
SC_tau = SC_tau.set_index('cellid').rename(columns={'value': 'SC_tau'})
# r0
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'
SC_r0 = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                ['region', 'siteid', 'cellid', 'value']]
SC_r0 = SC_r0.set_index('cellid').rename(columns={'value': 'SC_r0'})

# merge
SC_DF = pd.concat([SC_tau, SC_r0['SC_r0']], axis=1).copy()
SC_DF['dPCA_tau'] = np.nan
SC_DF['dPCA_r0'] = np.nan

# populatio values
ff_anal = DF.analysis == 'dPCA'
# tau
ff_param = DF.parameter == 'tau'
ff_source = DF.source == 'significance'
dPCA_tau = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
                ['region', 'siteid', 'value']]
dPCA_tau = dPCA_tau.set_index('siteid').rename(columns={'value': 'dPCA_tau'})

# r0
ff_param = DF.parameter == 'r0'
ff_source = DF.source == 'dprime'
dPCA_r0 = DF.loc[ff_anal & ff_probe & ff_trans & ff_param & ff_source,
               ['region', 'siteid', 'value']]
dPCA_r0 = dPCA_r0.set_index('siteid').rename(columns={'value': 'dPCA_r0'})

# merge
dPCA_DF = pd.concat([dPCA_tau, dPCA_r0['dPCA_r0']], axis=1)

# apply population values to single SC_DF
for cellid, row in SC_DF.iterrows():
    site = row['siteid']
    SC_DF.loc[cellid, 'dPCA_tau'] = dPCA_DF.loc[site, 'dPCA_tau']
    SC_DF.loc[cellid, 'dPCA_r0'] = dPCA_DF.loc[site, 'dPCA_r0']



# filter out awnomalous data
ff_r0 = SC_DF['SC_r0'] >= 0.2
ff_tau = SC_DF['SC_tau'] <= 2000

toplot = SC_DF.loc[ff_tau & ff_r0, :]

x = 'dPCA_r0'
y = 'SC_r0'

fig, ax = plt.subplots()
ax = sns.regplot(x=x, y=y, data=toplot, color='black', ax=ax)
sns.despine(ax=ax)
left, right = ax.get_xlim()
ax.set_xlim(left, right + (right - left)/12)
_,_,r2,_,_ = sst.linregress(toplot[x], toplot[y])
_ = fplt.unit_line(ax, square_shape=False)

ax.set_xlabel(x, fontsize=ax_lab_size)
ax.set_ylabel(y, fontsize=ax_lab_size)
ax.tick_params(labelsize=ax_val_size)
ax.tick_params(labelsize=ax_val_size)

fig = ax.figure
fig.set_size_inches((6,6))
title = '{} vs {}, r={:.2f}'.format(x, y, r2)
fig.suptitle(title, fontsize=sub_title_size)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fplt.savefig(fig, 'DAC3_figures', title)
