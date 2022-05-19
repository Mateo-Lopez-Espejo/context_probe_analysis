import pathlib as pl
from configparser import ConfigParser
import joblib as jl

from src.root_path import config_path
from src.data.load import get_CPN_ids, get_runclass_ids, get_batch_ids

#### general configuration to import the right data and caches
config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220310_ctx_mod_metric_DF_tstat_cluster_mass_BS'
DF = jl.load(summary_DF_file)

# full CPN10_NTI A1 cellid set for model fitting
selected_sites = set(get_CPN_ids(10,'AllPermutations').siteid
                     ).intersection(set(get_runclass_ids("NTI").siteid)
                                    ).intersection(set(get_batch_ids(326).siteid))
filtered = DF.query("metric in ['integral'] and mult_comp_corr == 'bf_cp' and source == 'real' and "
                    "cluster_threshold == 0.05 and  value > 0 and "
                    f"site in {list(selected_sites)}")
cellid_A1_fit_set = set(filtered['id'].unique())


# full CPN10_NTI PEG cellid set for model fitting
selected_sites = set(get_CPN_ids(10,'AllPermutations').siteid
                     ).intersection(set(get_runclass_ids("NTI").siteid)
                                    ).intersection(set(get_batch_ids(327).siteid))
filtered = DF.query("metric in ['integral'] and mult_comp_corr == 'bf_cp' and source == 'real' and "
                    "cluster_threshold == 0.05 and  value > 0 and "
                    f"site in {list(selected_sites)}")
cellid_PEG_fit_set = set(filtered['id'].unique())


# subset 01: CPN10 + NTI, A1, top 20 highest mean(of instancese)-amplitude-integration neurons excluding non significant
selected_sites = set(get_CPN_ids(10,'AllPermutations').siteid
                     ).intersection(set(get_runclass_ids("NTI").siteid)
                                    ).intersection(set(get_batch_ids(326).siteid))
filtered = DF.query("metric in ['integral'] and mult_comp_corr == 'bf_cp' and source == 'real' and "
                    "cluster_threshold == 0.05 and  value > 0 and "
                    f"site in {list(selected_sites)}")
top_cells = filtered.groupby(['id', 'metric']).agg(mean_value=("value", 'mean'))
top_cells = top_cells.reset_index().sort_values(['mean_value'], ascending=False)
cellid_subset_01 = set(top_cells.head(20)['id'])

# subset 02: CPN10 + NTI + A1. top 20, INCLUDING non significant
selected_sites = set(get_CPN_ids(10,'AllPermutations').siteid
                     ).intersection(set(get_runclass_ids("NTI").siteid)
                                    ).intersection(set(get_batch_ids(326).siteid))
filtered = DF.query("metric in ['integral'] and mult_comp_corr == 'bf_cp' and source == 'real' and "
                    "cluster_threshold == 0.05 and "
                    f"site in {list(selected_sites)}")
top_cells = filtered.groupby(['id', 'metric']).agg(mean_value=("value", 'mean'))
top_cells = top_cells.reset_index().sort_values(['mean_value'], ascending=False)
cellid_subset_02 = set(top_cells.head(20)['id'])

########################################################################################################################
# site subsets

# bad sites for different reasons
bad_empirical = {'AMT031a', 'DRX008b', 'DRX021a', 'DRX023a', 'ley074a'}
no_permutations = {'ley058d'}
time_shift = {}
bad_sites = bad_empirical.union(no_permutations).union(time_shift)

# all good sites
good_sites = set(get_batch_ids(316).siteid)
good_sites = good_sites.difference(bad_sites)

########################################################################################################################
# maps neurons to batches. Makes sense since these are brain region dependent batches
batch_map = dict()
for bb in [326, 327]: # A1 and PEG batches
    cids = get_batch_ids(bb).cellid
    for cid in cids:
        batch_map[cid] = bb
