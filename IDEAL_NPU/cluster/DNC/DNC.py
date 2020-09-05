import time
import scipy.io as sio
import numpy as np
from scipy import sparse
import IDEAL_NPU.funs as Ifuns
from sklearn.cluster import KMeans
from .DNC_ import opt


class DNC(object):
    def __init__(self, X, c_true):
        self.X = X
        self.N = X.shape[0]
        self.time = 0
        self.c_true = c_true
        self.iter = 0

    def get_M(self, A, Da):
        Da[Da == 0] = 0.0000001
        Da_sq_inv = 1 / np.sqrt(Da)
        tmp = A * np.outer(Da_sq_inv, Da_sq_inv)

        tmp = np.maximum(tmp, tmp.T)

        tmp_sp = sparse.csr_matrix(tmp)
        ret = sparse.linalg.eigsh(tmp_sp, which='SA', k=1, return_eigenvectors=False)[0]

        eps = 2e-16
        if ret <= 0:
            lam = -ret + eps
        M = tmp + lam*np.eye(self.N)
        M = sparse.csr_matrix(M)
        return M

    def clu(self, graph_knn=0, graph_way="t_free", ITER=100):
        A = Ifuns.kng(self.X, knn=graph_knn, way=graph_way)
        d = np.sum(A, axis=1)

        M = self.get_M(A, d)

        A_csr = sparse.csr_matrix(A)
        y = KMeans(self.c_true, init="k-means++", n_init=1).fit(A_csr).labels_
        # y = np.random.randint(0, self.c_true, self.N)

        y = opt(M, d, y, self.c_true, ITER)
        return y
