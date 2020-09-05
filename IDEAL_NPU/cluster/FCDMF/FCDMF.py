import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans

from .FCDMF_ import opt
from IDEAL_NPU import funs as Ifuns


class FCDMF(object):
    def __init__(self, X, c_true):
        self.X = X
        self.N = X.shape[0]
        self.c_true = c_true

    def get_anchor(self, anchor_num, anchor_way):
        Anchor = Ifuns.get_anchor(X=self.X, m=anchor_num, way=anchor_way)
        return Anchor

    def get_graph(self, graph_knn, graph_way, Anchor):
        B = Ifuns.kng(X=self.X, knn=graph_knn, way=graph_way, Anchor=Anchor)
        return B

    def clu(self, KTIMES, ITER=100,
            anchor_num=0, anchor_way="k-means++",
            graph_knn=0, graph_way="t_free", km_init="k-means++",
            a1=0, a2=0):

        if anchor_num == 0:
            anchor_num = int(min(self.N / 2, 1024))
        Anchor = self.get_anchor(anchor_num=anchor_num, anchor_way=anchor_way)

        if graph_knn == 0:
            graph_knn = np.minimum(2 * self.c_true, anchor_num)
        B = self.get_graph(graph_knn=graph_knn, graph_way=graph_way, Anchor=Anchor)

        if a1 == 0:
            a1 = int(np.maximum(self.N/10/self.c_true, 1))
        if a2 == 0:
            a2 = int(np.maximum(self.N/10/self.c_true, 1))

        n, m = B.shape
        # initialize P
        B_sp = sparse.csr_matrix(B)
        U, S, VH = svds(B_sp, k=self.c_true, which="LM")

        ret = np.zeros((KTIMES, n))
        for ktimes in range(KTIMES):

            p = KMeans(self.c_true, init=km_init).fit(U[:, :self.c_true]).labels_

            V = VH.T
            q = KMeans(self.c_true, init=km_init).fit(V[:, :self.c_true]).labels_

            p = opt(p.astype(np.int32), q.astype(np.int32), B, self.c_true, a1=a1, a2=a2, ITER=ITER)

            ret[ktimes, :] = p

        return ret

    @property
    def ref(self):
        title = "Fast Clustering With Co-Clustering Via Discrete Non-Negative Matrix Factorization for Image Identification"
        return title
