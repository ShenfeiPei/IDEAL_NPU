import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances as EuDist2

from .WBKM_ import opt

from scipy import sparse


class WBKM(object):
    """
    obj(p, q) = ||D1^{-.5} X D2^{-.5} - D1^.5 P S Q.T D^.5||_F
    """
    def __init__(self, X, c_true):
        self.X = sparse.coo_matrix(X)
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.c_true = c_true

    def clu(self, ITER=30, km_init="k-means++"):

        eps = 2e-16

        # init
        d1 = np.sum(self.X, 1) + eps
        d1 = np.array(d1).reshape(-1)
        d2 = np.sum(self.X, 0) + eps
        d2 = np.array(d2).reshape(-1)

        p = KMeans(self.c_true, init=km_init).fit(self.X).labels_
        q = KMeans(self.c_true, init=km_init).fit(self.X.T).labels_
        # p = np.random.randint(0, self.c_true, self.N)
        # q = np.random.randint(0, self.c_true, self.d)

        np.ascontiguousarray(p, dtype=np.int32)
        np.ascontiguousarray(q, dtype=np.int32)
        np.ascontiguousarray(d1, dtype=np.float64)
        np.ascontiguousarray(d2, dtype=np.float64)
        # np.ascontiguousarray(self.X, dtype=np.float64)
        opt(p, q, d1, d2, self.X, self.c_true, ITER)

        return p

    @property
    def ref(self):
        title = "Weighted bilateral K-means algorithm for fast co-clustering and fast spectral clustering"
        return title
