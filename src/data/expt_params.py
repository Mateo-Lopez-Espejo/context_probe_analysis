import pandas as pd
import numpy as np
from src.data.load import get_CPN_ids


df = get_CPN_ids(10, 'AllPermutations')
df['sorted_sounds'] = df['Ref_SoundIndexes'].apply(lambda x: sorted(x))

arr = np.unique(np.asarray(df.Ref_SoundIndexes.tolist()),axis=0)

unique_sorted_seqs = [set(s) for s in arr.unique()]





