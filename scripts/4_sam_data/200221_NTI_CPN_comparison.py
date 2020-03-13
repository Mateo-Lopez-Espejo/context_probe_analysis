import numpy as np
from cpn_load import load


'''
since applying the dprime CPN analysis toe the NTI data was unsuccessfull, the next alternative to compare Sam and my
approach is to perform the CPN and NTI analysis to their respective datasets on recording sites that have both data
'''

#1. list sites with both datasets
    # list all NTI sites this have to be done manually
    # list all CPN sites, this should be trivial
    # check the intersection

#2. Calculates the dPrime for each site and all possible probes, context pairs and cells (?). This is the difficult part
# to summarize the outcome of all the

# 3. import and parse matlab results for Sam's NTI analysis. These results are in a cell by cell format, then it makes
# sense to calculate the dprimes idividually forP each cell