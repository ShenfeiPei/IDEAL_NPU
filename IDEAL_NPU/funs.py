import os
import random
import numpy as np
import scipy.io as sio
import pandas as pd
from scipy import stats

from coclust.evaluation.external import accuracy as acc_ori
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import adjusted_mutual_info_score as ami_ori
from sklearn.metrics import normalized_mutual_info_score as nmi_ori

from sklearn.metrics.pairwise import euclidean_distances as EuDist2
import time
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial


def get_anchor(X, m, way="random"):
    if way == "kmeans":
        A = KMeans(m, init='random').fit(X).cluster_centers_
    elif way == "kmeans2":
        A = KMeans(m, init='random').fit(X).cluster_centers_
        D = EuDist2(A, X)
        ind = np.argmin(D, axis=1)
        A = X[ind, :]
    elif way == "k-means++":
        A = KMeans(m, init='k-means++').fit(X).cluster_centers_
    elif way == "k-means++2":
        A = KMeans(m, init='k-means++').fit(X).cluster_centers_
        D = EuDist2(A, X)
        A = np.argmin(D, axis=1)

    elif way == "random":
        ids = random.sample(range(X.shape[0]), m)
        A = X[ids, :]
    return A


def precision(y_true, y_pred):
    assert (len(y_pred) == len(y_true))
    N = len(y_pred)
    y_df = pd.DataFrame(data=y_pred, columns=["label"])
    ind_L = y_df.groupby("label").indices
    ni_L = [stats.mode(y_true[ind]).count[0] for yi, ind in ind_L.items()]
    return np.sum(ni_L) / N


def recall(y_true, y_pred):
    re = precision(y_true=y_pred, y_pred=y_true)
    return re


def accuracy(y_true, y_pred):
    acc = acc_ori(true_row_labels=y_true, predicted_row_labels=y_pred)
    return acc


def ami(y_true, y_pred, average_method="max"):
    ret = ami_ori(labels_true=y_true, labels_pred=y_pred, average_method=average_method)
    return ret


def nmi(y_true, y_pred, average_method="max"):
    ret = nmi_ori(labels_true=y_true, labels_pred=y_pred, average_method=average_method)
    return ret


def load_Agg():
    this_directory = os.path.dirname(__file__)
    data_path = os.path.join(this_directory, "dataset/")
    name_full = os.path.join(data_path + "Agg.mat")
    X, y_true, N, dim, c_true = load_mat(name_full)
    return X, y_true, N, dim, c_true


def load_USPS():
    this_directory = os.path.dirname(__file__)
    data_path = os.path.join(this_directory, "dataset/")
    name_full = os.path.join(data_path + "USPS.mat")
    X, y_true, N, dim, c_true = load_mat(name_full)
    return X, y_true, N, dim, c_true


def load_mat(path):
    data = sio.loadmat(path)
    X = data["X"]
    y_true = data["Y"].astype(np.int32).reshape(-1)
    N, dim, c_true = X.shape[0], X.shape[1], len(np.unique(y_true))
    return X, y_true, N, dim, c_true


def save_mat(name_full, xy):
    sio.savemat(name_full, xy)


def matrix_index_take(X, ind_M):
    assert np.all(ind_M >= 0)

    n, k = ind_M.shape
    row = np.repeat(np.array(range(n), dtype=np.int32), k)
    col = ind_M.reshape(-1)
    ret = X[row, col].reshape((n, k))
    return ret


def matrix_index_assign(X, ind_M, Val):
    n, k = ind_M.shape
    row = np.repeat(np.array(range(n), dtype=np.int32), k)
    col = ind_M.reshape(-1)
    if isinstance(Val, (float, int)):
        X[row, col] = Val
    else:
        X[row, col] = Val.reshape(-1)


def kng(X, knn, way="gaussian", t="mean", Anchor=0):
    """
    :param X: data matrix of n by d
    :param knn: the number of nearest neighbors
    :param way: one of ["gaussian", "t_free"]
        "t_free" denote the method proposed in :
            "The constrained laplacian rank algorithm for graph-based clustering"
        "gaussian" denote the heat kernel
    :param t: only needed by gaussian, the bandwidth parameter
    :param Anchor: Anchor set, m by d
    :return: A, an sparse matrix (graph) of n by n if Anchor = 0 (default)
    """
    N, dim = X.shape
    if isinstance(Anchor, int):
        # n x n graph
        D = EuDist2(X, X, squared=True)
        ind_M = np.argsort(D, axis=1)
        if way == "gaussian":
            Val = matrix_index_take(D, ind_M[:, 1:(knn+1)])
            if t == "mean":
                t = np.mean(Val)
            elif t == "median":
                t = np.median(Val)
            Val = np.exp(-Val / t)
        elif way == "t_free":
            Val = matrix_index_take(D, ind_M[:, 1:(knn+2)])
            Val = Val[:, knn].reshape((-1, 1)) - Val[:, :knn]
            Val = Val / np.sum(Val, axis=1).reshape(-1, 1)
        A = np.zeros((N, N))
        matrix_index_assign(A, ind_M[:, 1:(knn+1)], Val)
        A = (A + A.T) / 2
    else:
        # n x m graph
        num_anchor = Anchor.shape[0]
        D = EuDist2(X, Anchor, squared=True)  # n x m
        ind_M = np.argsort(D, axis=1)
        if way == "gaussian":
            Val = matrix_index_take(D, ind_M[:, :knn])
            if t == "mean":
                t = np.mean(Val)
            elif t == "median":
                t = np.median(Val)
            Val = np.exp(-Val / t)
        elif way == "t_free":
            Val = matrix_index_take(D, ind_M[:, :(knn+1)])
            Val = Val[:, knn].reshape((-1, 1)) - Val[:, :knn]
            Val = Val / np.sum(Val, axis=1).reshape(-1, 1)
        A = np.zeros((N, num_anchor))
        matrix_index_assign(A, ind_M[:, :knn], Val)

    return A


def kmeans(X, c, rep, init="random", mini_batch=False, par=0, n_jobs=-1):
    """
    km = random or k-means++
    """
    if par == 0:
        Y = np.zeros((rep, X.shape[0]))
        if mini_batch:
            for i in range(rep):
                Y[i, :] = MiniBatchKMeans(n_clusters=c, init=init, n_init=1).fit(X).predict(X)
        else:
            for i in range(rep):
                Y[i, :] = KMeans(c, n_init=1, init=init).fit(X).labels_
    else:
        # parallel = Parallel(n_jobs=6)
        # if mini_batch:
        #     Y = parallel(delayed(minibatch_kmeans_par)(X, c, init, i) for i in range(rep))
        # else:
        #     Y = parallel(delayed(kmeans_par)(X, c, init, i) for i in range(rep))
        # Y = np.array(Y)
        pool = Pool(processes=6)

        partial_km = partial(kmeans_par, X=X, c=c, init=init)
        partial_km_batch = partial(minibatch_kmeans_par, X=X, c=c, init=init)
        rep_list = list(range(rep))
        print(mini_batch)
        if mini_batch:
            Y = pool.map(partial_km_batch, rep_list)
        else:
            Y = pool.map(partial_km, rep_list)
        Y = np.array(Y)

    return Y


def minibatch_kmeans_par(X, c, init, i):
    y = MiniBatchKMeans(n_clusters=c, init=init, n_init=1).fit(X).predict(X)
    return y


def kmeans_par(i, X, c, init):
    y = KMeans(c, n_init=1, init=init).fit(X).labels_
    return y


def relabel(y, offset=0):
    y_df = pd.DataFrame(data=y, columns=["label"])
    ind_dict = y_df.groupby("label").indices

    for yi, ind in ind_dict.items():
        y[ind] = offset
        offset += 1


def normalize_fea(fea, row):
    '''
    if row == 1, normalize each row of fea to have unit norm;
    if row == 0, normalize each column of fea to have unit norm;
    '''

    if 'row' not in locals():
        row = 1

    if row == 1:
        feaNorm = np.maximum(1e-14, np.sum(fea ** 2, 1).reshape(-1, 1))
        fea = fea / np.sqrt(feaNorm)
    else:
        feaNorm = np.maximum(1e-14, np.sum(fea ** 2, 0))
        fea = fea / np.sqrt(feaNorm)

    return fea

