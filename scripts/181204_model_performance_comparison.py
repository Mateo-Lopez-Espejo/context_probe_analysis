import nems.db as nd
import numpy as np
import matplotlib.pyplot as plt
import itertools as itt
import seaborn as sns
import scipy.stats as sst

batch = 310
results_file = nd.get_results_file(batch)

all_models = ['wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1', 'wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1',
              'wc.2x2.c-fir.2x15-lvl.1-stategain.S-dexp.1', 'wc.2x2.c-fir.2x15-lvl.1-dexp.1']

shortnames = {'resp': 'resp',
              'wc.2x2.c-fir.2x15-lvl.1-dexp.1': 'LN',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1': 'STP',
              'wc.2x2.c-fir.2x15-lvl.1-stategain.S-dexp.1': 'state',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1': 'STP_state'}


color_dict = {'resp': 'C0',
              'wc.2x2.c-fir.2x15-lvl.1-dexp.1': 'C1',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-dexp.1': 'C2',
              'wc.2x2.c-fir.2x15-lvl.1-stategain.S-dexp.1': 'C3',
              'wc.2x2.c-stp.2-fir.2x15-lvl.1-stategain.S-dexp.1': 'C4'}

voc_color = {'voc_{}'.format(cc): 'C{}'.format(cc) for cc in range(5)}
voc_cmpat = {'voc_0': 'Blues', 'voc_1': 'Oranges', 'voc_2': 'Greens', 'voc_3': 'Reds', 'voc_4': 'Purples'}

all_cells = nd.get_batch_cells(batch=310).cellid.tolist()


# filters only relevant models and  columns of the DF
results_file['site'] = [cellid.split('-')[0] for cellid in results_file.cellid]
ff_sites = results_file.site != 'BRT057b'
ff_modelname = results_file.modelname.isin(all_models)

filtered = results_file.loc[ff_modelname & ff_sites, ['cellid', 'modelname', 'r_test']]

pivoted = filtered.pivot(index='cellid', columns='modelname', values='r_test')
pivoted = pivoted.reset_index()

pivoted['site'] = [cellid.split('-')[0] for cellid in pivoted.cellid]

max_rval = np.nanmax(pivoted.loc[:, all_models].values)

for mod1, mod2 in itt.combinations(all_models, 2):
    fig, ax = plt.subplots()

    for ii, site in enumerate(pivoted.site.unique()):
        if site == 'BRT057b':
            continue
        ff_site = pivoted.site == site
        filt = pivoted.loc[ff_site, :]

        color = 'C{}'.format(ii)
        x = filt[mod1].values
        y = filt[mod2].values

        ax.scatter(x, y, color=color, label=site)

    ax.legend()
    ax.set_xlim(0, max_rval + 0.1)
    ax.set_ylim(ax.get_xlim())
    ax.plot(ax.get_xlim(), ax.get_xlim(), 'k--')
    ax.set_xlabel('{}\n{}'.format(shortnames[mod1], mod1))
    ax.set_ylabel('{}\n{}'.format(shortnames[mod2], mod2))
    plt.suptitle('model performance\nr_value')
    fig.set_size_inches(5,5)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_model_performance_{}_vs_{}.png'.format(shortnames[mod1], shortnames[mod2]), dpi=100)
    fig.savefig('/home/mateo/Pictures/DAC1/181205_model_performance_{}_vs_{}.svg'.format(shortnames[mod1], shortnames[mod2]))


tidy = filtered.replace(shortnames)
order = [short for short in shortnames.values() if short!='resp']
fig, ax = plt.subplots()
g = sns.barplot(x='modelname', y='r_test', data=tidy, order=order, ax=ax)
fig.set_size_inches(5, 5)
fig.suptitle('model performance summary\nWillcoxon test')
fig.savefig('/home/mateo/Pictures/DAC1/181205_sumary_model_performance.png', dpi=100)
fig.savefig('/home/mateo/Pictures/DAC1/181205_sumary_model_performance.svg')


pivi = tidy.pivot(index='cellid', columns='modelname', values='r_test')
for mod1, mod2 in itt.combinations(pivi.keys(), 2):
    x = pivi[mod1].values
    y = pivi[mod2].values
    w_test = sst.wilcoxon(x, y)
    print('{} vs {} pvalue: {:.3f}'.format(mod1, mod2, w_test.pvalue))

