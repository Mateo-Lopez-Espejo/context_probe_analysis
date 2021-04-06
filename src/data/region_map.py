from src.data.load import get_site_ids
from nems.db import pd_query
from warnings import warn


# get CPN sites
all_sites = get_site_ids(316)


region_map = dict()
for site in all_sites.keys():
    # pulls the region from celldb
    BF_querry = "select area from gCellMaster where siteid=%s"
    raw_area = pd_query(BF_querry, params=(site,)).iloc[0, 0]

    # Sanitizes region in case of missing values
    if raw_area is None:

        warn(f'site {site} has undefined region')
        print()
        continue
    else:
        area = raw_area.split(',')[0]
        if area == '':
            print(f'site {site} has undefined region')
            continue
        elif area not in ('A1', 'PEG'):
            print(f'site {site} has unrecognized region:{area}')
            continue

    region_map[site] = area
