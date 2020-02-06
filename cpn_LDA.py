import numpy as np
# from sklearn.discriminant_analysis import LDA

def _fit(X, N=1):
    '''
    gets the matrix mapping the given data to de defined number of LDA components
    :param X: ndarray with shape Clases x Observations x Dimensions
    :param N: int, number of components of the projection
    :return: ndarray with shape Dimensions x Components
    '''
    n_classes = X.shape[0]

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
    discrimination_axis = eig_vecs[:, 0:N]

    return discrimination_axis

def fit_over_time(X, N=1):
    '''
    calculates the LDA axis for an array with shape Categories x Observations x Dimensions x Time.
    The LDA is calculated independently for each time bin, attempting to discriminate between categories defined by the
    first dimension of the array
    :param X: arrays with shape Categories x Observations x Dimensions x Time.
    :param N: number of components of the LDA, by default 1
    :return: array with shape, Dimensions x Components x Time
    '''

    C, O, D, T = X.shape # Categories x Observations x Dimensions x Time.
    transformations = np.zeros([D, N, T])
    for tt in range(T):
        transformations[..., tt] = _fit(X[..., tt], N=N)

    return transformations

def transform_over_time(X, trans_vectors):
    '''
    Transforms the array X , in a timewise manner, using the collection of transformations of trans vectors.
    The transformation is performed in an arraywise manner for each dimension previous to the third to last dimension
    :param X: array with shape ... x Observations x Dimension x Time
    :param trans_vectors: array with shape ... x Dimensions x Components x Time
    :return: array with shape ... x Observations x Components x Time
    '''

    # the projected array should have the same shape as the input array except for the Dimensions/Components dimension
    newshape= list(X.shape)
    newshape[-2] = trans_vectors.shape[-2]

    projection = np.zeros(newshape)
    for tt in range(newshape[-1]):
        projection[...,tt] = np.matmul(X[..., tt], trans_vectors[..., tt])

    return projection

# wrappers for raster array

def _reorder_dims(array):
    '''
    swaps between 'standard' order and LDA compatible shape
    :param array: array with shape   Observations x Dimensions x Categories x Time
    :return: array with shape        Categories x Observations x Dimensions x Time.
    '''
    return np.transpose(array, axes=[2, 0, 1, 3])


def _recover_dims(array):
    '''
    swaps between LDA compatible shape and 'standard' shape
    :param array: array with shape  Categories x Observations x Dimensions x Time.
    :return: array with shape       Observations x Dimensions x Categories x Time
    '''
    return np.transpose(array, axes=[1, 2, 0, 3])


def fit_transform_over_time(array, N=1):
    '''
    Using the raw data in the input array, fits an LDA matrix to discriminate across categories over time.
    then transforms the matrix into the new low-dimensional space
    :param array: array with shape Observations x Dimensions x Categories x Time
    :return: projections: Transformed array with shape Observations x Components x Categories x Time;
             Transformations: trans matrixes with shape Dimensions x Components x Time
    '''

    # swaps dimensions into LDA compatible format
    array = _reorder_dims(array)

    transformations = fit_over_time(array, N=N)
    projections = _recover_dims(transform_over_time(array, transformations))

    return projections, transformations


