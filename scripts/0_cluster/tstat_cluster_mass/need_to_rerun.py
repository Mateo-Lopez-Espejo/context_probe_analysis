import pathlib as pl
from configparser import ConfigParser

import joblib as jl
from src.data.load import get_batch_ids
from src.root_path import config_path
from src.utils.subsets import bad_sites, all_cells

config = ConfigParser()
config.read_file(open(config_path / 'settings.ini'))

summary_DF_file = pl.Path(config['paths']['analysis_cache']) / f'220520_minimal_DF'

DF = jl.load(summary_DF_file)

proc_cells = set(DF.query("analysis == 'SC'").id.unique())

to_reprocess = all_cells.difference(proc_cells)
print(to_reprocess)
print(len(to_reprocess))

to_reprocess_sites = set(c.split('-')[0] for c in to_reprocess)
print(to_reprocess_sites)
print(len(to_reprocess_sites))

# check particularly bad ones
longs= list()
for bad in to_reprocess_sites:
    bb = set([c for c in proc_cells if c.split('-')[0] == bad])
    BB = set([c for c in all_cells if c.split('-')[0] == bad])

    print(BB.difference(bb))
    print(f"{len(BB.difference(bb))} of {len(BB)}")
    if len(BB.difference(bb)) > 10:
        longs.append(bad)

print(f'sites with more than 10 neurons missing: {longs}')


