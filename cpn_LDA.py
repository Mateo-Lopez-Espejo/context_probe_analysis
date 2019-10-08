import numpy as np
# from sklearn.discriminant_analysis import LDA

def _get_LDA_axis(x, y):
    '''
    x and y must be of dimensions: O x N, where O are observations and N are
    number of dimensions. For example, this could be trials x neurons
    '''
    n_classes = 2
    if x.shape[0] != y.shape[0]:
        if x.shape[0] < y.shape[0]:
            n = x.shape[0]
            idx = np.random.choice(np.arange(0, y.shape[0]), n, replace=False)
            y = y[idx, :]
        else:
            n = y.shape[0]
            idx = np.random.choice(np.arange(0, x.shape[0]), n, replace=False)
            x = x[idx, :]

    X = np.concatenate((x[np.newaxis, :, :], y[np.newaxis, :, :]), axis=0)

    # find best axis using LDA
    # STEP 1: compute mean vectors for each category
    mean_vectors = []
    for cl in range(0, n_classes):
        mean_vectors.append(np.mean(X[cl], axis=0))

    # STEP 2.1: Compute within class scatter matrix
    n_units = X.shape[-1]
    S_W = np.zeros((n_units, n_units))
    n_observations = X.shape[1]
    for cl, mv in zip(range(0, n_classes), mean_vectors):
        class_sc_mat = np.zeros((n_units, n_units))
        for r in range(0, n_observations):
            row, mv = X[cl, r, :].reshape(n_units, 1), mv.reshape(n_units, 1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat

    # STEP 2.2: Compute between class scatter matrix
    overall_mean = np.mean(X, axis=0).mean(axis=0)[:, np.newaxis]
    S_B = np.zeros((n_units, n_units))
    X_fl = X.reshape(-1, n_units)
    for i in range(X_fl.shape[0]):
        S_B += (X_fl[i, :].reshape(n_units, 1) - overall_mean).dot((X_fl[i, :].reshape(n_units, 1) - overall_mean).T)

    # STEP 3: Solve the generalized eigenvalue problem for the matrix S_W(-1) S_B
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.pinv(S_W).dot(S_B))
    if np.iscomplexobj(eig_vecs):
        eig_vals, eig_vecs = np.linalg.eigh(np.linalg.pinv(S_W).dot(S_B))
    #if np.any(eig_vals<0):
    #    import pdb; pdb.set_trace()
    # STEP 4: Sort eigenvectors and find the best axis (number of nonzero eigenvalues
    # will be at most number of categories - 1)
    sorted_idx = np.argsort(eig_vals)[::-1]
    eig_vecs = eig_vecs[:, sorted_idx]
    eig_vals = eig_vals[sorted_idx]

    # STEP 5: Project data onto the top axis
    discrimination_axis = eig_vecs[:, 0]

    return discrimination_axis


def get_LDA_ax_over_time(X ,Y):
    '''
    calculates the lda axis for a pair of arrays with shape Repetition x Cell x Time
    todo elaborate on documentation
    :param X:
    :param Y:
    :return:
    '''

    if X.shape != Y.shape:
        raise ValueError('both arrays should have the same shape: Reps x Cell x Time')

    R, C, T = X.shape
    all_axes = np.zeros([C,T])
    for tt in range(T):
        all_axes[:, tt] = _get_LDA_axis(X[:, :, tt], Y[:, :, tt])

    return  all_axes

def transform_over_time(X, trans_vectors):
    '''
    Transforms the array X , in a timewise manner, using the collection of transformations of trans vectors.
    Designed to get the 1-dim projection from LDA
    :param X: array with shape Repetitions x Cell x Time
    :param trans_vectors: array with shape Cell x Components x Time
    :return: array with shape Repetitions x Time
    '''
    if X.shape[1] != trans_vectors.shape[0] or X.shape[2] != trans_vectors.shape[1]:
        raise ValueError('arrays have inconsistent shapes')

    R, C, T = X.shape
    all_trans = np.zeros([R,T])

    for tt in range(T):
        all_trans[:,tt] = np.matmul(X[:, :, tt], trans_vectors[..., tt])

    return all_trans

















