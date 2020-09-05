cimport numpy as np
import numpy as np
np.import_array()

from scipy import sparse

def opt(M, d, y, c, ITER):
    y = _opt(M, d, y, c, ITER)
    return y

cdef get_F(double[::1] d, double[::1] d_sq, int[::1] y, int c):

    cdef int N = y.shape[0]
    cdef double eps = 2.2204e-16
    cdef int[::1] one2N = np.arange(0, N, dtype=np.int32)

    cdef double[::1] ydy = np.zeros(c)
    cdef double[::1] F_v = np.zeros(N)

    # update F -> G
    cdef int i = 0
    cdef int tmp_c = 0

    ydy = np.zeros(c)
    for i in range(N):
        tmp_c = y[i]
        ydy[tmp_c] += d[i]

    for i in range(N):
        tmp_c = y[i]
        F_v[i] = d_sq[i]/(np.sqrt(ydy[tmp_c]+eps))

    F = sparse.coo_matrix((F_v, (one2N, y)), shape=(N, c))
    F = sparse.csc_matrix(F)
    return F


cdef update_y(double[::1] d, double[::1] d2, G, y):
    N, c = G.shape
    eps = 2e-16

    eye_c = np.eye(c, dtype=np.int32)
    Y = eye_c[y, :]

    D2G = np.diag(d2)@G
    tmp_yg = np.diag(Y.T.dot(D2G))
    cdef np.ndarray[double, ndim=1, mode="c"] yg = np.ascontiguousarray(tmp_yg, dtype=np.float64)
    tmp_ydy = np.diag(Y.T@np.diag(d)@Y + eps * eye_c)
    cdef np.ndarray[double, ndim=1, mode="c"] ydy = np.ascontiguousarray(tmp_ydy, dtype=np.float64)

    ind = np.argsort(d)
    cdef int flag = 0
    cdef int Iter = 0

    for Iter in range(N):
        flag = 0
        for tmp_i in range(N):
            i = ind[tmp_i]
            c_old = y[i]

            N1 = yg + D2G[i, :]*(1 - Y[i, :])
            DE1 = ydy + d[i]*(1-Y[i, :])

            N2 = yg - D2G[i, :]*Y[i, :]
            DE2 = ydy - d[i]*Y[i, :]

            c_new = np.argmax(N1/np.sqrt(DE1) - N2/np.sqrt(DE2))

            if c_new != c_old:
                Y[i, c_old] = 0
                Y[i, c_new] = 1
                y[i] = c_new

                yg[c_old] -= D2G[i, c_old]
                yg[c_new] += D2G[i, c_new]
                ydy[c_old] -= d[i]
                ydy[c_new] += d[i]
                flag = 1

        if flag==0:
            break
    return y


cdef _opt(M, d, y, int c, int ITER):
    cdef int N = y.shape[0]
    cdef int i = 0
    cdef double eps = 2e-16

    cdef double[::1] d_sq = np.zeros(N)
    d_sq = np.sqrt(d)

    F = get_F(d, d_sq, y, c)
    G = M@F

    cdef np.ndarray[double, ndim=1, mode="c"] obj = np.zeros(ITER)

    for Iter in range(ITER):
        y = update_y(d, d_sq, G, y)
        F = get_F(d, d_sq, y, c)
        G = M@F

        Obj_M = F.T@G
        obj[Iter] = np.trace(Obj_M.toarray())

        if Iter > 1 and np.abs((obj[Iter] - obj[Iter-1])/obj[Iter]) < 1e-10:
            break
        if Iter > 29:
            a = obj[(Iter-9):(Iter-4)] - obj[(Iter-4):(Iter+1)]
            if np.sum(np.abs(a)) < 1e-10:
                break

    return y





