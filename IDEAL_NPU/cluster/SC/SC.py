import numpy as np
import time
from scipy import sparse
import IDEAL_NPU.funs as Ifuns


class SC(object):
    # Ncut_Ng
    def __init__(self, X, c_true):
        self.X = X
        self._time = 0
        self.c_true = c_true

    def emb(self, A):
        d = np.sum(A, 1)
        d[d == 0] = 0.0000001
        d_inv = 1 / np.sqrt(d)
        tmp = A * np.outer(d_inv, d_inv)
        tmp = np.maximum(tmp, tmp.T)

        tmp_sp = sparse.csr_matrix(tmp)
        ret = sparse.linalg.eigsh(tmp_sp, which='LA', k=self.c_true)[1]

        return ret

    def clu(self, graph_knn=0, graph_way="t_free", km_rep=1, km_init="random"):
        t1 = time.time()
        A = Ifuns.kng(self.X, knn=graph_knn, way=graph_way)
        F = self.emb(A)
        F = F / (np.sqrt(np.sum(F ** 2, 1)).reshape(-1, 1))

        y = Ifuns.KMeans(n_clusters=self.c_true, init=km_init, n_init=km_rep).fit(F).labels_
        self._time = time.time() - t1
        return y

    @property
    def time(self):
        return self._time
