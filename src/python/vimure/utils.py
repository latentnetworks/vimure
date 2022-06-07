import sklearn

import numpy as np
import sktensor as skt

from sktensor.sptensor import fromarray

def match_arg(x, lst):
    return [el for el in lst if x == el]

def calculate_average_over_reporter_mask(X, R):
    if not isinstance(R, np.ndarray):
        R = R.toarray()
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    Xavg = np.einsum("lijm,lijm->lij", X, R)
    norm = R.sum(axis=-1)
    Xavg[norm > 0] /= norm[norm > 0]

    return Xavg


def calculate_AUC(pred, data0, mask=None):
    """
    Return the AUC of the link prediction. It represents the probability that a randomly chosen missing connection
    (true positive) is given a higher score by our method than a randomly chosen pair of unconnected vertices
    (true negative).

    Parameters
    ----------
    pred : ndarray
           Inferred values.
    data0 : ndarray
            Given values.
    mask : ndarray
           Mask for selecting a subset of the adjacency tensor.

    Returns
    -------
    AUC value.
    """

    data = (data0 > 0).astype("int")
    if mask is None:
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(data.flatten(), pred.flatten())
    else:
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(data[mask > 0], pred[mask > 0])

    return sklearn.metrics.auc(fpr, tpr)


def calculate_overall_reciprocity(Y):
    return np.logical_and(Y > 0, np.transpose(Y, axes=(1, 0)) > 0).sum() / Y.sum()


def get_item_array_from_subs(A, ref_subs):
    """
    Get values of ref_subs entries of a dense tensor.
    Output is a 1-d array with dimension = number of non zero entries.
    """

    return np.array(
        [
            A[a, i, j, m][0] if isinstance(A[a, i, j, m], np.ndarray) else A[a, i, j, m]
            for a, i, j, m in zip(*ref_subs)
        ]
    )


def is_sparse(X):
    """
    Check whether the input tensor is sparse.
    It implements a heuristic definition of sparsity. A tensor is considered sparse if:
    given
    M = number of modes
    S = number of entries
    I = number of non-zero entries
    then
    N > M(I + 1)

    Parameters
    ----------
    X : ndarray
        Input data.

    Returns
    -------
    Boolean flag: true if the input tensor is sparse, false otherwise.
    """

    M = X.ndim
    S = X.size
    I = X.nonzero()[0].size

    return S > (I + 1) * M


def sptensor_from_dense_array(X):
    """
    Create an sptensor from a ndarray or dtensor.

    Parameters
    ----------
    X : ndarray
        Input data.

    Returns
    -------
    sptensor from a ndarray or dtensor.
    """

    subs = X.nonzero()
    vals = X[subs]

    return skt.sptensor(subs, vals, shape=X.shape, dtype=X.dtype)


def sptensor_from_list(X):
    """
    Create an sptensor a sptensor from a list.

    Assuming it is a list of dimensions L x M with sparse matrices as elements
    """

    if isinstance(X, skt.sptensor):
        M = X.shape[3]
        L = X.shape[0]
    else:
        M = len(X)
        L = len(X[0])
    N = None  # Discover N later, during the loops
    Xdtype = None  # Discovery Xdtype later, during the loops

    subs_tot = []
    vals = []
    c = 0
    for l in range(L):
        for m in range(M):
            if not isinstance(X[m][l], list):
                if N is None:
                    N = X[m][l].shape[0]
                    Xdtype = X[m][l].dtype

                subs_ij = X[m][l].nonzero()
                I = subs_ij[0].shape[0]
                subs_l = l * np.ones(I).astype("int")
                subs_m = m * np.ones(I).astype("int")

                vals.extend(X[m][l].data)

                if c == 0:
                    subs_tot = [subs_l, subs_ij[0], subs_ij[1], subs_m]
                else:
                    subs_tot[0] = np.concatenate((subs_tot[0], subs_l))
                    subs_tot[1] = np.concatenate((subs_tot[1], subs_ij[0]))
                    subs_tot[2] = np.concatenate((subs_tot[2], subs_ij[1]))
                    subs_tot[3] = np.concatenate((subs_tot[3], subs_m))
                c += 1

    vals = np.array(vals)
    subs = tuple(subs_tot)
    assert subs[0].shape == vals.shape

    return skt.sptensor(subs, vals, shape=(L, N, N, M), dtype=Xdtype)


def sparse_max(A, B):
    """
    Return the element-wise maximum of sparse matrices `A` and `B`.
    """

    AgtB = (A > B).astype(int)
    M = AgtB.multiply(A - B) + B

    return M


"""
UTIL functions related to VimureModel
"""


def get_optimal_threshold(model):
    return 0.54 * model.G_exp_nu - 0.01


def apply_rho_threshold(model, threshold=None):
    """
    Apply a threshold to binarise the rho matrix and return the recovered Y
    """
    if threshold is None:
        threshold = get_optimal_threshold(model)

    Y_rec = np.copy(model.rho_f[:, :, :, 1])
    Y_rec[Y_rec < threshold] = 0
    Y_rec[Y_rec >= threshold] = 1
    return Y_rec


def preprocess(X):
    """
    Pre-process input data tensor.
    If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.

    Parameters
    ----------
    X : ndarray/list
        Input data.

    Returns
    -------
    X : sptensor/dtensor
        Pre-processed data. If the input is sparse, returns an int sptensor. Otherwise, returns an int dtensor.
    """

    if isinstance(X, skt.sptensor) or isinstance(X, skt.dtensor):
        return X
    elif isinstance(X, list):
        X = sptensor_from_list(X)
    else:
        if not X.dtype == np.dtype(int).type:
            X = X.astype(int)
        if isinstance(X, np.ndarray) and is_sparse(X):
            X = fromarray(X)
        else:
            X = skt.dtensor(X)

    return X
