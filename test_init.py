import cpp_epochs as cpe
import joblib as jl
import cpp_PCA as cpca

test_rec_path = '/home/mateo/context_probe_analysis/pickles/BRT037b'
loaded_rec = jl.load(test_rec_path)

rec = cpe.set_recording_subepochs(loaded_rec, set_pairs=True)
sig = rec['resp']
eps = sig.epochs

rec_pca, pca_stats = cpca.recording_PCA(rec, signal_names='resp', inplace=False, method='scikit', center=False)

matrix = rec_pca['resp_PCs'].extract_epoch('C0_P1')

