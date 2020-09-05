import numpy as np
from ._agci_emb import emb
import IDEAL_NPU.funs as Ifuns


class AGCI(object):
    def __init__(self, X, c_true):
        self.c_true = c_true
        self.X, self.ind, self.ind_re = np.unique(X, return_index=True, return_inverse=True, axis=0)

    def get_anchor(self, m=0, way="random"):
        if m == 0:
            m = int(min(self.X.shape[0] / 2, 1024))

        Anchor = Ifuns.get_anchor(self.X, m, way=way)
        return Anchor

    def get_graph(self, knn, Anchor, way="t_free"):
        Z = Ifuns.kng(self.X, knn=knn, way=way, Anchor=Anchor)
        return Z

    # verified
    #                                   verified                      verified
    def clu(self, anchor_num=0, anchor_way="random", graph_knn=0, graph_way="t_free", emb_ITER=100, km_times=1, km_init="random"):
        Anchor = self.get_anchor(m=anchor_num, way=anchor_way)
        Z = self.get_graph(graph_knn, Anchor, way=graph_way)
        F = emb(Z, self.c_true, ITER=emb_ITER)
        y = Ifuns.KMeans(n_clusters=self.c_true, init=km_init, n_init=km_times).fit(F).labels_
        # Y = Ifuns.kmeans(F, self.c_true, rep=km_times, init=km_init)
        # Y2 = [y[self.ind_re] for y in Y]
        # Y = np.array(Y2)
        return y[self.ind_re]

    @property
    def ref(self):
        title = "Fast Spectral Clustering for Unsupervised Hyperspectral Image Classification"
        return title
