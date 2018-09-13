import cpp_epochs as cpe
import joblib as jl
import cpp_PCA as cpca
import matplotlib.pyplot as plt
import numpy as np

test_rec_path = '/home/mateo/context_probe_analysis/pickles/BRT037b'
loaded_rec = jl.load(test_rec_path)

rec = cpe.set_recording_subepochs(loaded_rec, set_pairs=True)
sig = rec['resp']
eps = sig.epochs

# normalize matrix
full_mat = sig.rasterize()._data
mean = np.mean(full_mat.T, axis=0)
full_mat = (full_mat.T - mean).T

folded_mat = sig.rasterize().extract_epoch('C0_P1')

sk_sig, pca_sig = cpca.signal_PCA(sig, center=True)
sk_rec, pca_rec = cpca.recording_PCA(rec, signal_names='all', inplace=False, center=True)

U = pca_sig.components_ # 2d array nComponents x nFeatures
V = sk_sig._data

U = pca_rec['resp'].components_
V = sk_rec['resp_PCs']._data


#### plotting

# Transformation matrix
fig, ax = plt.subplots()

ax.imshow(U)
ax.set_ylabel('components')
ax.set_xlabel('cells')

# data vs principal component

time = -1
PC = 17
cell = 9
fig, ax = plt.subplots()

ax.plot(full_mat[cell, :time], label='cell') # original data
ax.plot(V[PC,:time], label='projection')
ax.legend()
ax.set_ylabel('spike rate')
ax.set_xlabel('time')


# end of the story, canned PCA works wonders, from the output ht

