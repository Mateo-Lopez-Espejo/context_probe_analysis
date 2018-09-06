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


# rec_pca, pca_stats = cpca.recording_PCA(rec, signal_names='resp', inplace=False, method='scikit', center=False)

sk_sig, sk_pca = cpca.signal_PCA(sig, center=True)

char_sig, char_pca = cpca.charlie_PCA(sig, center=True)


U = dict()
U['sklearn'] = sk_pca.components_
U['charlie'] = char_pca['loading'].T

V = dict()
V['sklearn'] = sk_sig._data
V['charlie'] = char_sig._data



fig, axes = plt.subplots(1,2)

axes = np.ravel(axes)

for ii, (key, val) in enumerate(U.items()):
    ax = axes[ii]
    ax.imshow(val)
    ax.set_title(key)
    if key == 'sklearn':
        ax.set_ylabel('components')
        ax.set_xlabel('chanels')



time = 100
PC = 1
channel = 6
fig, ax = plt.subplots()

ax.plot(full_mat[channel, :time])
for source, projection in V.items():
    ax.plot(projection[0,:time], label=source)
    ax.legend()











