import pathlib as pl
from configparser import ConfigParser
import joblib as jl

from src.root_path import config_path
from src.data.load import get_CPN_ids, get_runclass_ids, get_batch_ids

#### general configuration to import the right data and caches
config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))
remote = True

########################################################################################################################
# site subsets

# bad sites for different reasons
bad_empirical = {'AMT031a', 'DRX008b', 'DRX021a', 'DRX023a', 'ley074a'}
no_permutations = {'ley058d'}
time_shift = {}
bad_sites = bad_empirical.union(no_permutations).union(time_shift)

# all good sites
if remote:
    print("cannot connect to database, loading hardcoded sites")
    good_sites = {'AMT020a', 'AMT021b', 'AMT026a', 'AMT028b', 'AMT029a', 'AMT030a', 'AMT032a', 'ARM005e',
                  'ARM017a', 'ARM018a', 'ARM019a', 'ARM021b', 'ARM022b', 'ARM023a', 'ARM024a', 'ARM025a',
                  'ARM026b', 'ARM027a', 'ARM028b', 'ARM029a', 'ARM031a', 'ARM032a', 'ARM033a', 'CRD002a',
                  'CRD003b', 'CRD004a', 'CRD005b', 'CRD011c', 'CRD012b', 'CRD014b', 'CRD018d', 'CRD019b',
                  'TNC006a', 'TNC008a', 'TNC009a', 'TNC010a', 'TNC011a', 'TNC012a', 'TNC013a', 'TNC014a',
                  'TNC015a', 'TNC016a', 'TNC017a', 'TNC018a', 'TNC019a', 'TNC020a', 'TNC021a', 'TNC022a',
                  'TNC023a', 'TNC024a', 'TNC028a', 'TNC029a', 'TNC043a', 'TNC044a', 'TNC045a', 'TNC047a',
                  'TNC048a', 'TNC049a', 'TNC050a', 'TNC051a', 'TNC062a', 'ley070a', 'ley072b', 'ley075b'}

else:
    good_sites = set(get_batch_ids(316).siteid)

good_sites = good_sites.difference(bad_sites)

########################################################################################################################
# this will get less neurons as it also runs a filter on isolation: >= 95%
if remote:
    print("cannot connect to database, loading hardcoded cells. TODO!")
else:
    all_cells = set(get_batch_ids(316).query(f"siteid not in {list(bad_sites)}").cellid)

# permutation sites with enough reps, carefull, as this does not filter by isolation
# cell_10 = set(get_CPN_ids(10,'AllPermutations').query(f"siteid not in {list(bad_sites)}").cellid)

# full CPN10_NTI A1 cellid set for model fitting
# all_A1_cells = set(get_batch_ids(326).cellid) # no isolation filter
# cellid_A1_fit_set = all_A1_cells.intersection(cell_10).intersection(all_cells)


# full CPN10_NTI PEG cellid set for model fitting
# all_PEG_cells = set(get_batch_ids(327).cellid)
# cellid_PEG_fit_set = all_PEG_cells.intersection(cell_10).intersection(all_cells)


########################################################################################################################
# smaller subsets used to prototype models

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220310_ctx_mod_metric_DF_tstat_cluster_mass_BS'
DF = jl.load(summary_DF_file)

# subset 01: CPN10 + NTI, A1, top 20 highest mean(of instancese)-amplitude-integration neurons excluding non significant
# selected_sites = set(get_CPN_ids(10,'AllPermutations').siteid
#                      ).intersection(set(get_runclass_ids("NTI").siteid)
#                                     ).intersection(set(get_batch_ids(326).siteid))
# filtered = DF.query("metric in ['integral'] and mult_comp_corr == 'bf_cp' and source == 'real' and "
#                     "cluster_threshold == 0.05 and  value > 0 and "
#                     f"site in {list(selected_sites)}")
# top_cells = filtered.groupby(['id', 'metric']).agg(mean_value=("value", 'mean'))
# top_cells = top_cells.reset_index().sort_values(['mean_value'], ascending=False)
# cellid_subset_01 = set(top_cells.head(20)['id'])

# subset 02: CPN10 + NTI + A1. top 20, INCLUDING non significant
# selected_sites = set(get_CPN_ids(10,'AllPermutations').siteid
#                      ).intersection(set(get_runclass_ids("NTI").siteid)
#                                     ).intersection(set(get_batch_ids(326).siteid))
# filtered = DF.query("metric in ['integral'] and mult_comp_corr == 'bf_cp' and source == 'real' and "
#                     "cluster_threshold == 0.05 and "
#                     f"site in {list(selected_sites)}")
# top_cells = filtered.groupby(['id', 'metric']).agg(mean_value=("value", 'mean'))
# top_cells = top_cells.reset_index().sort_values(['mean_value'], ascending=False)
# cellid_subset_02 = set(top_cells.head(20)['id'])



########################################################################################################################
# maps neurons to batches. Makes sense since these are brain region dependent batches
batch_map = dict()
for bb in [326, 327]: # A1 and PEG batches
    cids = get_batch_ids(bb).cellid
    for cid in cids:
        batch_map[cid] = bb
