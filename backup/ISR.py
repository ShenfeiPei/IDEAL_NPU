print("Warning: ISR has been abandoned.")
import time
import scipy.io as sio
import numpy as np
from scipy import sparse
import IDEAL_NPU.funs as Ifuns

from .ISR_ import get_M, update_y


class ISR(object):
    def __init__(self, X, c_true):
        self.X = X
        self.N = X.shape[0]
        self.time = 0
        self.c_true = c_true

    def get_graph(self, knn, way):
        A = Ifuns.kng(self.X, knn=knn, way=way)
        return A

    def emb(self, A):
        d = np.sum(A, 1)
        d[d == 0] = 0.0000001
        d_inv = 1 / np.sqrt(d)
        tmp = A * np.outer(d_inv, d_inv)
        tmp = np.maximum(tmp, tmp.T)

        tmp_sp = sparse.csr_matrix(tmp)
        ret = sparse.linalg.eigsh(tmp_sp, which='LA', k=self.c_true)[1]
        return ret

    def sr(self, Da, F, ITER):
        # initialize
        tmp = np.sqrt(np.diagonal(F@F.T))
        tmp = F/tmp.reshape(-1, 1)
        # y = np.argmax(np.abs(tmp), axis=1)
        y = np.random.randint(0, self.c_true, self.N)

        for iter in range(ITER):

            # R, G
            # ydy_sq_inv = np.diag(np.diag(Y.T@np.diag(Da)@Y + 2.2204e-16)**(-0.5))
            # print(ydy_sq_inv)
            M = get_M(y.astype(np.int32), self.c_true, Da)

            # print(M[:4, :4])
            U, S, Vh = np.linalg.svd(M@F)
            R = Vh@U.T
            G = F@R

            y = np.argmax(G, axis=1)
            # Y = eye_c[tmp, :]
            # [~, g] = max(FQ, [], 2);
            # Gsr4 = TransformL(g, class_num);
            # update Y
            y_old = y.copy()
            y = update_y(G, y.astype(np.int32), self.c_true, Da)

            if np.sum(np.abs(y-y_old)) == 0:
                break
        return y

    def clu(self, graph_knn=0, graph_way="t_free", sr_ITER=10):
        A = self.get_graph(knn=graph_knn, way=graph_way)
        sio.savemat("D:/hh.mat", {"A": A})
        Da = np.sum(A, axis=1)
        F = self.emb(A)
        y = self.sr(Da, F, ITER=sr_ITER)

        return y
