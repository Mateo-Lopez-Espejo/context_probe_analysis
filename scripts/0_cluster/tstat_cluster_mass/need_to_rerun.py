import pathlib as pl
from configparser import ConfigParser

import joblib as jl
from src.data.load import get_batch_ids
from src.root_path import config_path
from src.utils.subsets import bad_sites

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220520_minimal_DF'

DF = jl.load(summary_DF_file)

proc_cells = set(DF.query("analysis == 'SC'").id.unique())

all_cells = set(get_batch_ids(316).query(f"siteid not in {list(bad_sites)}").cellid.unique())

to_reprocess = all_cells.difference(proc_cells)

to_reprocess = set(c.split('-')[0] for c in to_reprocess)

print(to_reprocess)
