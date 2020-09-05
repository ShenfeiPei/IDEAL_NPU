import time
import warnings
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans


class SSC(object):
    def __init__(self, X, c_true, alpha=0.01):
        self.NUMERICAL_ZERO = 1e-16
        self.N = X.shape[0]
        self.dim = X.shape[1]
        self.c_true = c_true
        self._time = 0

        t1 = time.time()
        self.X = X / np.sqrt(np.sum(X ** 2, axis=1)).reshape(-1, 1)
        tmp = np.sum(self.X, axis=0).reshape(-1, 1)
        self.deg = (self.X@tmp).reshape(-1) - 1

        ind1_out = np.where(self.deg < self.NUMERICAL_ZERO)[0]
        ind1_in = np.where(self.deg >= self.NUMERICAL_ZERO)[0]

        n = int(np.round(self.N*alpha) - len(ind1_out))

        tmp_ind = np.argpartition(self.deg[ind1_in], n)[0:n]
        ind2_out = ind1_in[tmp_ind]

        self.ind_out = np.concatenate((ind1_out, ind2_out))
        self.ind_in = np.setdiff1d(np.arange(self.N), self.ind_out)

        self.deg_inv_in = 1/np.sqrt(self.deg[self.ind_in])
        self._time += time.time() - t1

    def emb(self):
        X2 = self.X[self.ind_in, :]*self.deg_inv_in.reshape(-1,1)
        max_c = np.min(X2.shape) - 1
        if self.c_true > max_c:
            warnings.warn("First dim vectors are used instead of c, since max_c < c_true")
            U, S, _ = svds(X2, max_c)
        else:
            U, S, _ = svds(X2, self.c_true)
        return U, S

    def Ncut(self, U):
        U = U[:, 1:]*self.deg_inv_in.reshape(-1, 1)
        return U

    def DM(self, U, S, t):
        U = U[:, 1:]*self.deg_inv_in.reshape(-1, 1)
        if t>0:
            lam = S[1:]**2 - np.mean(self.deg_inv_in)
            lam = lam**t
            U = U*lam.reshape(1, -1)
        return U

    def km(self, U, km_init, km_rep):
        eye_c = np.eye(self.c_true)
        tmp_y = np.zeros(self.N, dtype=np.int32)

        y = KMeans(self.c_true, n_init=km_rep, init=km_init).fit(U).labels_
        y = y.reshape(-1).astype(np.int32)

        Y = eye_c[y, :]
        cen = (self.X[self.ind_in, :].T.dot(Y)).T
        cen = cen/np.sum(Y, axis=0).reshape(-1, 1)

        sim = self.X[self.ind_out, :].dot(cen.T)
        y2 = np.argmax(sim, axis=1).astype(np.int32)

        tmp_y[self.ind_in] = y
        tmp_y[self.ind_out] = y2
        return tmp_y

    def clu(self, km_rep, km_init="random", way="NJW", dm_t=0):
        t1 = time.time()

        U, S = self.emb()
        if way == "Ncut":
            U = self.Ncut(U)
        if way == "DM":
            U = self.DM(U, S, dm_t)

        if way not in ["DM", "Ncut", "NJW"]:
            print("error")

        U = U/np.sqrt(np.sum(U**2, 1)).reshape(-1, 1)
        self._time += time.time() - t1

        t1 = time.time()
        ret = self.km(U, km_init=km_init, km_rep=km_rep)
        self._time += (time.time() - t1)/km_rep

        return ret

    @property
    def time(self):
        return self._time

    @property
    def ref(self):
        title = "Scalable spectral clustering with cosine similarity, 2018, ICPR"
        return title
