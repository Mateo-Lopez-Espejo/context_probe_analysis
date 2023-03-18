from src.data.load import get_batch_ids
from nems.db import pd_query
from warnings import warn

remote = True

# get CPN sites
all_sites = get_batch_ids(316).siteid.unique().tolist()

if remote:
    region_map = {'TNC013a': 'A1', 'ARM023a': 'PEG', 'TNC050a': 'PEG', 'TNC012a': 'A1', 'TNC047a': 'PEG',
                  'TNC048a': 'PEG', 'TNC009a': 'A1', 'AMT029a': 'PEG', 'ARM028b': 'PEG', 'TNC008a': 'A1',
                  'CRD012b': 'PEG', 'TNC021a': 'A1', 'ARM027a': 'PEG', 'ARM022b': 'PEG', 'TNC020a': 'A1',
                  'ARM033a': 'A1', 'ARM026b': 'PEG', 'TNC045a': 'A1', 'ley070a': 'A1', 'CRD005b': 'A1', 'TNC017a': 'A1',
                  'TNC043a': 'A1', 'AMT026a': 'A1', 'CRD011c': 'PEG', 'TNC014a': 'A1', 'TNC010a': 'A1',
                  'ARM025a': 'PEG', 'CRD019b': 'A1', 'TNC028a': 'PEG', 'AMT032a': 'PEG', 'ARM018a': 'PEG',
                  'AMT028b': 'PEG', 'ARM024a': 'PEG', 'TNC023a': 'PEG', 'TNC006a': 'A1', 'CRD002a': 'A1',
                  'TNC051a': 'PEG', 'TNC029a': 'PEG', 'AMT021b': 'A1', 'ley072b': 'A1', 'ARM031a': 'A1',
                  'CRD003b': 'A1', 'AMT020a': 'A1', 'ARM005e': 'A1', 'TNC015a': 'A1', 'CRD018d': 'A1', 'ARM019a': 'PEG',
                  'CRD014b': 'PEG', 'AMT030a': 'PEG', 'TNC016a': 'A1', 'TNC044a': 'A1', 'ARM029a': 'A1',
                  'TNC062a': 'A1', 'TNC022a': 'PEG', 'CRD004a': 'A1', 'ARM021b': 'PEG', 'ARM032a': 'A1',
                  'TNC011a': 'A1', 'TNC018a': 'A1', 'TNC024a': 'PEG', 'TNC019a': 'A1', 'ley075b': 'A1',
                  'TNC049a': 'PEG', 'ARM017a': 'PEG'}
else:
    region_map = dict()
    for site in all_sites:
        # pulls the region from celldb
        BF_querry = "select area from gCellMaster where siteid=%s"
        raw_area = pd_query(BF_querry, params=(site,)).iloc[0, 0]

        # Sanitizes region in case of missing values
        if raw_area is None:
            warn(f'\nsite {site} has undefined region\n')
            continue
        else:
            area = raw_area.split(',')[0]
            if area == '':
                print(f'importing region_map: site {site} has undefined region')
                continue
            elif area not in ('A1', 'PEG'):
                print(f'importing region_map: site {site} has unrecognized region:{area}')
                continue

        region_map[site] = area
