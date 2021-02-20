import pathlib as pl
from nems_lbhb.penetration_map import penetration_map
import matplotlib.pyplot as plt

sites = ['ARM012d', 'ARM013b', 'ARM014b', 'ARM015b', 'ARM016c', 'ARM017a', 'ARM018a', 'ARM019a', 'ARM020a',
         'ARM021b', 'ARM022b', 'ARM023a', 'ARM024a', 'ARM025a', 'ARM026b', 'ARM027a', 'ARM028b', 'ARM029a', 'ARM030a',
         'ARM031a', 'ARM032a', 'ARM033a', 'ARM034a']

# fig, coords = penetration_map(sites, cubic=False, flip_X=True, flatten=False)
fig, coords = penetration_map(sites, equal_aspect=True, flip_X=True, flatten=True)

# saves the scatter
mappath = pl.Path('/auto/users/mateo/Pictures/Armillaria/ARM_LH_map.png')
fig.savefig(mappath, transparent=True)