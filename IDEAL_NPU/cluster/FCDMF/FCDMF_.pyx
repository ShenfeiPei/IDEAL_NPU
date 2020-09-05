cimport numpy as np
import numpy as np
np.import_array()

def update_PQ(
        np.ndarray[double, ndim=2, mode="c"] D,
        np.ndarray[int, ndim=1, mode="c"] y,
        int L):
    cdef int n = D.shape[0]
    cdef int c = D.shape[1]
    cdef int flag = 1

    cdef np.ndarray[int, ndim=1, mode="c"] nc = np.zeros(c, dtype=np.int32)
    for i in range(n):
        nc[y[i]] += 1

    cdef int c_old = 0
    cdef int c_new = 0
    cdef double d_old = 0

    while flag == 1:
        flag = 0
        for i in range(n):
            c_old = y[i]
            if nc[c_old] > L:

                d_old = D[i, c_old]
                c_new = c_old
                for j in range(c):
                    if D[i, j] < d_old:
                        d_old = D[i, j]
                        c_new = j

                if c_new != c_old:
                    y[i] = c_new
                    nc[c_old] -= 1
                    nc[c_new] += 1
                    flag = 1

def update_S(np.ndarray[int, ndim=1, mode="c"] p,
             np.ndarray[int, ndim=1, mode="c"] q,
             np.ndarray[double, ndim=2, mode="c"] B, int c,
             np.ndarray[double, ndim=2, mode="c"] S):

    cdef int n = B.shape[0]
    cdef int m = B.shape[1]

    for i in range(c):
        for j in range(c):
            S[i, j] = 0

    for i in range(n):
        for j in range(m):
            S[p[i], q[j]] += B[i, j]

    cdef np.ndarray[int, ndim=1, mode="c"] pnc = np.zeros(c, dtype=np.int32)
    for i in range(n):
        pnc[p[i]] += 1

    cdef np.ndarray[int, ndim=1, mode="c"] qnc = np.zeros(c, dtype=np.int32)
    for i in range(m):
        qnc[q[i]] += 1

    for i in range(c):
        for j in range(c):
            S[i, j] /= pnc[i]*qnc[j]


def opt(np.ndarray[int, ndim=1, mode="c"] p,
        np.ndarray[int, ndim=1, mode="c"] q,
        np.ndarray[double, ndim=2, mode="c"] B, int c, int a1, int a2, int ITER):

    cdef int n = B.shape[0]
    cdef int m = B.shape[1]
    cdef int tmp_c = 0

    cdef np.ndarray[double, ndim=2, mode="c"] BT = np.zeros((m, n))
    BT = B.T.copy()

    cdef np.ndarray[double, ndim=2, mode="c"] S = np.zeros((c, c))
    cdef np.ndarray[double, ndim=2, mode="c"] SQT = np.zeros((c, m))
    cdef np.ndarray[double, ndim=2, mode="c"] STPT = np.zeros((c, n))

    cdef np.ndarray[double, ndim=1, mode="c"] vn = np.zeros(n)
    cdef np.ndarray[double, ndim=1, mode="c"] vm = np.zeros(m)
    cdef np.ndarray[double, ndim=1, mode="c"] vc = np.zeros(c)

    cdef np.ndarray[double, ndim=2, mode="c"] M_nc = np.zeros((n, c))
    cdef np.ndarray[double, ndim=2, mode="c"] M_mc = np.zeros((m, c))

    cdef np.ndarray[double, ndim=2, mode="c"] DBP = np.zeros((n, c))
    cdef np.ndarray[double, ndim=2, mode="c"] DBQ = np.zeros((m, c))

    cdef np.ndarray[double, ndim=1, mode="c"] obj = np.zeros(ITER)

    for Iter in range(ITER):
        update_S(p, q, B, c, S)

        # SQT, cxm
        for i in range(m):
            # SQT[:, i] = S[:, q[i]]
            tmp_c = q[i]
            for j in range(c):
                SQT[j, i] = S[j, tmp_c]

        EuDist2(B, vn, SQT, vc, M_nc, DBP)
        update_PQ(DBP, p, a1)

        # STPT, cxn
        for i in range(n):
            # STPT[:, i] = ST[:, p[i]]
            tmp_c = p[i]
            for j in range(c):
                STPT[j, i] = S[tmp_c, j]


        EuDist2(BT, vm, STPT, vc, M_mc, DBQ)
        update_PQ(DBQ, q, a2)

        obj[Iter] = 0
        for i in range(n):
            for j in range(m):
                obj[Iter] += (B[i, j] - S[p[i], q[j]])**2

        if Iter > 2 and (obj[Iter] - obj[Iter - 1]) / obj[Iter - 1] < 1e-6:
            break

    return p

# C = Eudist2(A, B)
# va[i] = ||Ai||_2^2
def EuDist2(np.ndarray[double, ndim=2, mode="c"] A,
            np.ndarray[double, ndim=1, mode="c"] va_,
            np.ndarray[double, ndim=2, mode="c"] B,
            np.ndarray[double, ndim=1, mode="c"] vb_,
            np.ndarray[double, ndim=2, mode="c"] ABT_,
            np.ndarray[double, ndim=2, mode="c"] C):

    A_norm(A, va_)
    A_norm(B, vb_)
    ABT_ = A.dot(B.T)

    cdef int n = A.shape[0]
    cdef int m = B.shape[0]

    for i in range(n):
        for j in range(m):
            C[i, j] = va_[i] + vb_[j] - 2*ABT_[i, j]


def A_norm(np.ndarray[double, ndim=2, mode="c"] A,
            np.ndarray[double, ndim=1, mode="c"] va):

    cdef int n = A.shape[0]
    cdef int d = A.shape[1]

    for i in range(n):
        va[i] = 0
        for j in range(d):
            va[i] += A[i, j]*A[i, j]
