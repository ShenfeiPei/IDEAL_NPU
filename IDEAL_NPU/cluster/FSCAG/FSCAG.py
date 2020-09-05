import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from IDEAL_NPU import funs as Ifuns


class FSCAG(object):
    def __init__(self, X, c_true):
        self.X = X
        self.N = X.shape[0]
        self.c_true = c_true

    def clu(self, anchor_num=0, anchor_way="random",
            graph_knn=0, graph_way="t_free", km_init="k-means++"):

        if anchor_num == 0:
            anchor_num = int(min(self.N / 2, 1024))
        if graph_knn == 0:
            graph_knn = np.minimum(2 * self.c_true, anchor_num)

        Anchor = Ifuns.get_anchor(X=self.X, m=anchor_num, way=anchor_way)
        Z = Ifuns.kng(X=self.X, knn=graph_knn, way=graph_way, Anchor=Anchor)
        B = Z / np.sqrt(np.sum(Z, axis=0)).reshape(1, -1)

        # initialize P
        B_sp = sparse.csr_matrix(B)
        U, S, VH = svds(B_sp, k=self.c_true + 1, which="LM")
        p = KMeans(self.c_true, init=km_init).fit(U[:, 1:]).labels_

        return p

    @property
    def ref(self):
        title = "Fast Spectral Clustering With Anchor Graph for Large Hyperspectral Images"
        return title
