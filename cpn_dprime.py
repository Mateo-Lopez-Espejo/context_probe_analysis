import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import pandas as pd
from itertools import permutations
import scipy.fftpack as fp
import scipy.signal as ss
import logging


def dprime(array0, array1, absolute=True):
    '''
    calculates the unidimensional timewise dprime between two two-dimensional arrays with shape Trial x Time
    :param array0: ndarray
    :param array1: ndarray
    :return: 1D array with shape Time
    '''

    if absolute is True:
        dprime = (np.abs(np.mean(array0, axis=0) - np.mean(array1, axis=0)) /
                  np.sqrt(0.5 * (np.var(array0, axis=0) + np.var(array1, axis=0))))
    elif absolute is False:
        dprime = (np.mean(array0, axis=0) - np.mean(array1, axis=0) /
                  np.sqrt(0.5 * (np.var(array0, axis=0) + np.var(array1, axis=0))))
    else:
        raise ValueError(f'absolute must be bool but is {type(absolute)}')



    # check for edge cases
    if np.any(np.isnan(dprime)):

        for ec in np.where(np.isnan(dprime))[0]:

            if ((array0[:, ec].mean() - array1[:, ec].mean()) != 0) & \
                    ((np.var(array0[:, ec]) + np.var(array1[:, ec])) == 0):

                #print("Inf. case")
                dprime[ec] = abs(array0[:, ec].mean() - array1[:, ec].mean())

            elif ((array0[:, ec].mean() - array1[:, ec].mean()) == 0) & \
                    ((np.var(array0[:, ec]) + np.var(array1[:, ec])) == 0):

                dprime[ec] = 0

            else:
                raise SystemError('WTF?')

    return dprime


def ndim_dprime(array0, array1, absolute=True):
    '''
    calculates the multidimensional timewise dprime between two tri-dimentional arrays with shape Trial x Cell x Time
    the multidimentiona dprime is defined as the euclidean distance
    :param array0: ndarray
    :param array1: ndarray
    :return: 1D array with shape time
    '''

    # iterates over each neuron and calculates the d'
    all_dprimes = list()
    for cell in range(array0.shape[1]):
        this_dprime = dprime(array0[:, cell, :], array1[:, cell, :], absolute=absolute)
        all_dprimes.append(this_dprime)
    all_dprimes = np.stack(all_dprimes)

    # claculates the ndim hypotenuse fromm each cell d' (per time bin)
    ndim_dprime = np.sqrt(np.sum(np.square(all_dprimes), axis=0))

    return ndim_dprime


def _compute_dprime(rec, units=None, epochs=None, mean_resp=None, verbose=False,
                    LDA=True, return_subspace_overlap=False):
    r = rec.copy()
    r_all_data = rec.copy()
    r_all_data = r_all_data.create_mask(True)
    r_all_data = r_all_data.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    r = r.apply_mask(reset_epochs=True)
    r = r.and_mask(['PreStimSilence', 'PostStimSilence'], invert=True)
    r = r.apply_mask(reset_epochs=True)
    if epochs is None:
        epochs = np.unique([str(ep) for ep in r.epochs.name if 'STIM' in ep]).tolist()
    else:
        # mask the recording to contain only matched epochs
        r = r.and_mask(epochs)
        r = r.apply_mask(reset_epochs=True)
        r_all_data = r_all_data.and_mask(epochs)
        r_all_data = r_all_data.apply_mask(reset_epochs=True)

    all_data_overall_dict = r_all_data['resp'].extract_epochs(epochs)
    overall_dict = r['resp'].extract_epochs(epochs)
    stim_dict = r['stim'].rasterize().extract_epochs(epochs)
    mean_dict = {k: np.mean(v, axis=0) for (k, v) in overall_dict.items()}
    if mean_resp is not None:
        print("using user supplied mean response dictionary")
        mean_dict = mean_resp # used for if you'd like to pass overall means,
                              # rather than mask-dependent means
    if units is not None:
        overall_dict = {k: np.take(v, units, axis=1) for (k, v) in overall_dict.items()}
        mean_dict = {k: np.take(v, units, axis=0) for (k, v) in mean_dict.items()}
        all_data_overall_dict = {k: np.take(v, units, axis=1) for (k, v) in all_data_overall_dict.items()}

    samp = mean_dict[epochs[0]]
    nSegs = samp.shape[-1] * len(epochs)
    segs_per_epoch = samp.shape[-1]
    if return_subspace_overlap:
        idx = [str(i[0])+'_'+str(i[1]) for i in list(permutations(range(0, nSegs), 2))]
        DP = pd.DataFrame(index=idx, columns=['dprime', 'dec_vs_PC1', 'dec_vs_PC2',
        'dec_vs_PC1_all', 'dec_vs_PC2_all', 'PC1_var', 'PC1_var_all', 'PC2_var', 'PC2_var_all',
        'stim_cos_dist', 'stim_abs_dist',
        'noise_mag'])
    else:
        DP = np.nan * np.ones((nSegs, nSegs))
    idx = 0
    # loop over all epochs (outer loops over the current segment)
    for i, ep in enumerate(epochs):
        # loop over all segments in this epoch
        for seg in range(segs_per_epoch):
            # for this epoch / seg compute dprime vs. all others
            cur_seg_mean = mean_dict[ep][:, seg]
            cur_seg_st = overall_dict[ep][:, :, seg]
            cur_seg_st_all_data = all_data_overall_dict[ep][:, :, seg]
            cur_stim = stim_dict[ep][0, :, seg]
            # save of all dprimes for this single segment (will be length:
            # number of other segments).
            dp = []
            # loop over all other sound segments
            other_idx = 0
            for j, ep2 in enumerate(epochs):
                # loop over bins in current comparison epochs
                count = 0
                for bin in range(segs_per_epoch):
                    # for each comparison bin we compute dprime
                    if (ep2==ep) & (bin==seg):
                        other_idx += 1
                        pass
                    else:
                        # compute dprime between the two stimuli
                        u1 = cur_seg_mean
                        u2 = mean_dict[ep2][:, bin]
                        # get projections of each single trial (dot product of single trial
                        # response onto the decoding axis)
                        #if (idx == 23) and (other_idx==24):
                        #    import pdb; pdb.set_trace()
                        if len(cur_seg_st.squeeze().shape) > 1:
                            if LDA:
                                # TO-DO: This should happen over all trials, not
                                # just big/small pupil trials. So, we use the overall
                                # data for this
                                d_unit = get_LDA_axis(cur_seg_st_all_data,
                                            all_data_overall_dict[ep2][:, :, bin])
                                #d_unit = get_LDA_axis(cur_seg_st, overall_dict[ep2][:, :, bin])
                            else:
                                d_unit = get_null_axis(cur_seg_st_all_data,
                                            all_data_overall_dict[ep2][:, :, bin])
                                #d = u2 - u1
                                #d_unit = unit_vector(d)

                            stim1 = np.matmul(cur_seg_st, d_unit)
                            stim2 = np.matmul(overall_dict[ep2][:, :, bin], d_unit)
                        else:
                            stim1 = cur_seg_st
                            stim2 = overall_dict[ep2][:, :, bin]

                        dprime = (abs(stim1.mean() - stim2.mean())) / np.sqrt(0.5 * (np.var(stim1) + np.var(stim2)))

                        # check for weird cases
                        if ((abs(stim1.mean() - stim2.mean())) > 0) & \
                                ((np.var(stim1) + np.var(stim2)) == 0):
                                print("Inf. case")
                                dprime = abs(stim1.mean() - stim2.mean())
                        elif ((abs(stim1.mean() - stim2.mean())) == 0) & \
                                ((np.var(stim1) + np.var(stim2)) == 0):
                                dprime = 0
                        #elif np.iscomplexobj(stim1):
                        #    import pdb; pdb.set_trace()
                        else:
                            pass

                        if return_subspace_overlap:
                            # calculate the similarity (cos dist) between principal
                            # component 1 of residuals and the decoding
                            # axis (do it w/in condition and across)

                            # w/in condition first:
                            cur_seg_res = cur_seg_st - cur_seg_st.mean(0)
                            other_seg_res = overall_dict[ep2][:, :, bin] - overall_dict[ep2][:, :, bin].mean(0)
                            residuals = np.concatenate((cur_seg_res, other_seg_res), axis=0)
                            pca = PCA()
                            pca.fit(residuals)
                            noise_axis = pca.components_[0]
                            noise_var1 = pca.explained_variance_ratio_[0]
                            noise_axis = noise_axis / np.linalg.norm(noise_axis)
                            similarity1 = np.dot(noise_axis, d_unit)

                            noise_axis = pca.components_[1]
                            noise_var2 = pca.explained_variance_ratio_[1]
                            noise_axis = noise_axis / np.linalg.norm(noise_axis)
                            similarity2 = np.dot(noise_axis, d_unit)

                            total_var = np.sum(pca.singular_values_**2)

                            # across all conditions:
                            cur_seg_res_all = cur_seg_st_all_data - cur_seg_st_all_data.mean(0)
                            other_seg_res_all = all_data_overall_dict[ep2][:, :, bin] - all_data_overall_dict[ep2][:, :, bin].mean(0)
                            residuals = np.concatenate((cur_seg_res_all, other_seg_res_all), axis=0)
                            pca = PCA()
                            pca.fit(residuals)
                            noise_axis = pca.components_[0]
                            noise_var1_all = pca.explained_variance_ratio_[0]
                            noise_axis = noise_axis / np.linalg.norm(noise_axis)
                            similarity1_all = np.dot(noise_axis, d_unit)

                            noise_axis = pca.components_[1]
                            noise_var2_all = pca.explained_variance_ratio_[1]
                            noise_axis = noise_axis / np.linalg.norm(noise_axis)
                            similarity2_all = np.dot(noise_axis, d_unit)

                            total_var_all = np.sum(pca.singular_values_**2)

                            # also calculate the cosine distance between the two
                            # stimuli (means)
                            s1 = cur_seg_st_all_data.mean(0)
                            s2 = all_data_overall_dict[ep2][:, :, bin].mean(0)
                            stim_cos_dist = np.dot(s1 / np.linalg.norm(s1),
                                                   s2 / np.linalg.norm(s2))

                            # and calculate the difference in their magnitudes,
                            stim_abs_dist = abs(np.linalg.norm(s1) - np.linalg.norm(s2))
                            # normalized by the variance along the overall mean
                            uStim = np.concatenate((s1[:, np.newaxis], s2[:, np.newaxis]), axis=1).mean(axis=-1)
                            st_all = np.concatenate((cur_seg_st_all_data,
                                                    all_data_overall_dict[ep2][:, :, bin]), axis=0)
                            proj = np.matmul(st_all, uStim / np.linalg.norm(uStim))
                            std = np.std(proj)
                            stim_abs_dist /= std


                            id = str(idx) + '_' + str(other_idx)
                            DP.loc[id]['dprime'] = dprime
                            DP.loc[id]['dec_vs_PC1'] = abs(similarity1)
                            DP.loc[id]['dec_vs_PC2'] = abs(similarity2)
                            DP.loc[id]['dec_vs_PC1_all'] = abs(similarity1_all)
                            DP.loc[id]['dec_vs_PC2_all'] = abs(similarity2_all)
                            DP.loc[id]['stim_cos_dist'] = abs(stim_cos_dist)
                            DP.loc[id]['PC1_var'] = noise_var1
                            DP.loc[id]['PC1_var_all'] = noise_var1_all
                            DP.loc[id]['PC2_var'] = noise_var2
                            DP.loc[id]['PC2_var_all'] = noise_var2_all
                            DP.loc[id]['stim_abs_dist'] = stim_abs_dist
                        else:
                            DP[idx, other_idx] = dprime

                        other_idx += 1  #iterate the "other" idx
            idx +=1 # iterate the index
    return DP


##### functions from charlie

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def get_LDA_axis_sklearn(x, y):
    '''
    x and y must be of dimensions: O x N, where O are observations and N are
    number of dimensions. For example, this could be trials x neurons
    '''
    n_classes = 2
    n_units = x.shape[1]
    n_dims = x.shape[-1]
    if x.shape[0] != y.shape[0]:
        if x.shape[0] < y.shape[0]:
            n = x.shape[0]
            idx = np.random.choice(np.arange(0, y.shape[0]), n, replace=False)
            y = y[idx, :]
        else:
            n = y.shape[0]
            idx = np.random.choice(np.arange(0, x.shape[0]), n, replace=False)
            x = x[idx, :]
    else:
        n = x.shape[0]

    X = np.concatenate((x[np.newaxis, :, :], y[np.newaxis, :, :]), axis=0)
    import pdb; pdb.set_trace()
    clf = LinearDiscriminantAnalysis(n_components=n_classes)
    clf.fit(X.reshape(-1, n_units), np.vstack((np.zeros(n), np.ones(n))).reshape(-1))

    axis = clf.coef_ / np.linalg.norm(clf.coef_.squeeze())

    return axis.squeeze()

def get_null_axis(x, y):
    '''
    Return unit vector from centroid of x to centroid of y
    x and y must be of dimensions: O x N, where O are observations and N are
    number of dimensions. For example, this could be trials x neurons
    '''
    ux = x.mean(axis=0)
    uy = y.mean(axis=0)

    d = ux - uy

    return unit_vector(d)

def get_LDA_axis(x, y):
    '''
    x and y must be of dimensions: O x N, where O are observations and N are
    number of dimensions. For example, this could be trials x neurons
    '''
    n_classes = 2
    n_dims = x.shape[-1]
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