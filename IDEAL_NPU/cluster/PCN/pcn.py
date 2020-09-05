import time
import numpy as np
import numba as nb
nb.config.NUMBA_DEFAULT_NUM_THREADS = 6
import warnings
from scipy import sparse
from scipy.sparse.csgraph import connected_components


class PCN(object):
    def __init__(self, NN, NND):
        self.N, self.knn = NN.shape
        self.NN = NN
        self.NND = NND
        self.time = 0
        self.ind_sorted = np.arange(self.N*self.knn)
        self.C = np.zeros((self.N, self.knn), dtype=np.int32) - 1

    # NN n x k, NND nxk (except self)
    def sorting(self):
        aux_ind = np.arange(self.N*self.knn)
        self.ind_sorted = np.lexsort((aux_ind, self.NND.reshape(-1)))

    def compact(self):
        self.C = compact_(self.NN, self.C, self.ind_sorted)

    def compute_W(self):
        W = compute_W_(self.NN, self.C)
        return W

    def density_f(self, rho):
        density = density_f_(self.NN, self.C, rho)
        return density

    def clu_f(self, density):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = clu_f_(self.NN, self.C, density)
        return y

    def cluster(self):
        t_start = time.time()
        self.sorting()
        self.compact()
        W = self.compute_W()
        rho = np.sum(W, axis=1)
        density = self.density_f(rho)
        y = self.clu_f(density)
        t_end = time.time()
        self.time = t_end - t_start
        return y

    def get_time(self):
        return self.time


@nb.jit(nopython=True)
def compact_(NN, C, ind_sorted):
    N, knn = NN.shape
    for ind in ind_sorted:
        i = ind // knn
        k1 = ind % knn
        j = NN[i, k1]
        if i > j:
            if k1 == 0 or C[i, k1-1] >= 0:
                k2 = -1
                for ii in range(knn):
                    if NN[j, ii] == i:
                        k2 = ii
                        break
                if k2 >= 0:
                    if k2 == 0 or C[j, k2-1] >= 0:
                        C[i, k1] = k2
                        C[j, k2] = k1
    return C


@nb.jit(nopython=True)
def compute_Wij(NN, i, k1, j):
    N, knn = NN.shape
    sum1 = 1
    for nbi in range(k1):
        tmp_nb = NN[i, nbi]
        for ii in range(knn):
            if NN[j, ii] == tmp_nb:
                sum1 += 1
                break
    return sum1


@nb.jit(nopython=True, parallel=True)
def compute_W_(NN, C):
    N, knn = C.shape
    W = np.zeros((N, knn), dtype=np.float64)
    for i in range(N):
        for k1 in range(knn):
            if C[i, k1] >= 0:
                j = NN[i, k1]
                if j < i:
                    k2 = C[i, k1]
                    sum1 = compute_Wij(NN, i, k1, j)
                    sum2 = compute_Wij(NN, j, k2, i)
                    W[i, k1] = (sum1 + sum2) / min(k1+1, k2+1)
                    W[j, k2] = W[i, k1]
            else:
                break

    return W


@nb.jit(nopython=True)
def density_f_(NN, C, rho):
    N, knn = NN.shape
    density = np.zeros(N, dtype=np.int32)

    max_count = np.max(rho)
    rho = np.floor(rho / max_count * 100)
    thr_g = np.median(rho[rho > 0])
    for i in range(N):
        if rho[i] >= thr_g:
            density[i] = 2

    for i in range(N):
        if C[i, 0] < 0 or density[i] == 2:
            continue

        ni = 0
        for k in range(knn):
            if C[i, k] >= 0:
                ni += 1
            else:
                break

        local_rhos = rho[NN[i, :ni]]
        ind = np.where(local_rhos < thr_g)[0]
        local_rhos = local_rhos[ind]
        if len(local_rhos) <= 1:
            density[i] = 1
            continue

        thr = np.median(local_rhos)
        if rho[i] >= thr:
            density[i] = 1

    return density


@nb.jit(nopython=False)
def clu_f_(NN, C, density):
    N, knn = NN.shape
    row = list()
    col = list()
    for i in range(N):
        if density[i] == 0:
            for k1 in range(knn):
                if C[i, k1] >= 0:
                    j = NN[i, k1]
                    if density[j] > 0:
                        row.append(i)
                        col.append(j)
                        break
                else:
                    break
        else:
            for k1 in range(knn):
                if C[i, k1] >= 0:
                    j = NN[i, k1]
                    if density[j] > 0:
                        row.append(i)
                        col.append(j)
                    if density[j] == 0:
                        break
                else:
                    break

    data = list(np.ones(len(row)))
    graph = sparse.coo_matrix((data, (row, col)), shape=(N, N), copy=True)
    graph.eliminate_zeros()
    ci, y = connected_components(csgraph=graph, directed=False, return_labels=True)
    return y
