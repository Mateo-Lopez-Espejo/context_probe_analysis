import nems.db as nd
import pandas as pd
import pathlib as pl

all_rec_ids = set()
all_sites = list()
all_regions = list()
for batch, region in zip([318, 319], ['A1', 'PEG']):
    df = nd.get_batch_cell_data(batch=batch)
    files = df.parm.unique()
    rec_ids = [file.split('/')[-1].split('.')[0] for file in files]
    sites = [rec_id.split('_')[0][:7] for rec_id in rec_ids]

    all_rec_ids.update(rec_ids)
    all_sites.extend(sites)
    all_regions.extend([region] * len(rec_ids))

region_map = pd.DataFrame({'site_id': all_sites, 'region': all_regions})

empirical_bad = {'DRX008b14_p_NTI', 'CRD002a16_p_NTI'}
good_rec_ids = all_rec_ids.difference(empirical_bad)
all_animals = set([rec[:3] for rec in good_rec_ids])
print(good_rec_ids)
print(all_animals)

mapfile = pl.Path('/auto/users/mateo/integration_quilt/scrambling-ferrets/rasters/region_map.csv')
region_map.to_csv(mapfile)