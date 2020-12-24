from src.data.load import get_site_ids
from warnings import warn
# get CPN sites
all_sites = get_site_ids(316)

region_map = {'AMT020a': 'A1',
              'AMT026a': 'PEG',
              'AMT028b': 'PEG',
              'AMT029a': 'PEG',
              'AMT030a': 'PEG',
              'AMT031a': 'PEG',
              'AMT032a': 'PEG',
              'CRD002a': 'A1',
              'CRD003b': 'A1',
              'CRD004a': 'A1',
              'CRD011c': 'PEG',
              'CRD012b': 'PEG',
              'CRD014b': 'PEG',
              'CRD018d': 'A1',
              'CRD019b': 'A1',
              'DRX008b': 'A1',
              'DRX021a': 'A1',
              'DRX023a': 'A1',
              'ley070a': 'A1',
              'ley072b': 'A1',
              'ley074a': 'A1',
              'ley075b': 'A1'}

difference = set(all_sites.keys()).difference(set(region_map.keys()))
if difference:
    warn(f'{difference} have not been mapped to regions')