cimport numpy as np
import numpy as np
np.import_array()

import time

cimport cython
from cython.parallel import prange

from scipy import sparse


@cython.boundscheck(False)
@cython.wraparound(False)
def opt(int[::1] p, int[::1] q, double[::1] d1, double[::1] d2, X, int c, int ITER):

    cdef int n = X.shape[0]
    cdef int d = X.shape[1]

    # X2 = D1-1 X D2-1
    X2 = C_get_X2(X, d1, d2)
    X2_csc = sparse.csc_matrix(X2)
    X2_csr = sparse.csr_matrix(X2)

    cdef double[::1] X2_norm = np.zeros(n)
    cdef double[::1] X2T_norm = np.zeros(d)
    C_L2norm_sparseA(X2, 1, 1, X2_norm)
    C_L2norm_sparseA(X2, 0, 1, X2T_norm)

    cdef double[::1] vc = np.zeros(c)
    cdef double[::1] pdp = np.zeros(c)
    cdef double[::1] qdq = np.zeros(c)
    cdef double[::1] s = np.zeros(c)

    cdef np.ndarray[double, ndim=2, mode="c"] SQT = np.zeros((c, d))
    cdef double[:, ::1] SQT_view = SQT
    cdef np.ndarray[double, ndim=2, mode="c"] PS = np.zeros((n, c))
    cdef double[:, ::1] PS_view = PS
    cdef np.ndarray[double, ndim=2, mode="c"] DXP = np.zeros((n, c))
    cdef np.ndarray[double, ndim=2, mode="c"] DXQ = np.zeros((d, c))

    cdef int[::1] p_old = np.zeros(n, dtype=np.int32)

    cdef int tmp_c = 0
    cdef int i = 0
    cdef double eps = 2e-16
    cdef int flag = 0
    cdef int Iter = 0

    for Iter in range(ITER):
        p_old = p

        # update S
        C_sum_vy(d1, p, c, pdp)
        C_sum_vy(d2, q, c, qdq)
        C_get_s(X, p, q, c, pdp, qdq, eps, s)

        # update p
        for i in range(c):
            vc[i] = 0

        for i in range(d):
            tmp_c = q[i]
            SQT_view[tmp_c, i] = s[tmp_c]
            vc[tmp_c] += s[tmp_c]*s[tmp_c]

        C_EuDist2(X2_csr, X2_norm, 1, SQT, vc, 1, 1, DXP)
        tmpp = np.argmin(DXP, axis=1)
        p = tmpp.astype(np.int32)

        # update q
        for i in range(c):
            vc[i] = 0
        for i in range(n):
            tmp_c = p[i]
            PS_view[i, tmp_c] = s[tmp_c]
            vc[tmp_c] += s[tmp_c]*s[tmp_c]

        C_EuDist2(X2_csc, X2T_norm, 1, PS, vc, 1, 0, DXQ)
        tmpq = np.argmin(DXQ, axis=1)
        q = tmpq.astype(np.int32)

        flag = 0
        for i in range(n):
            if p_old[i] != p[i]:
                flag = 1
                break
        if flag == 0:
            break
    return p

@cython.boundscheck(False)
@cython.wraparound(False)
# X2 = D1-1 X D2-1
cdef C_get_X2(X, double[::1] d1, double[::1] d2):

    # cdef int n = X.shape[0]
    # cdef int d = X.shape[1]
    # cdef int i = 0
    # cdef int j = 0
    # for i in prange(n, nogil=True):
    #     for j in range(d):
    #         X2[i, j] = X[i, j]/(d1[i]*d2[j])

    cdef int N = X.nnz
    cdef int i = 0
    cdef double[::1] X2_data = np.zeros(N)
    cdef int r = 0
    cdef int c = 0
    for i in range(N):
        r = X.row[i]
        c = X.col[i]
        X2_data[i] = X.data[i]/(d1[r]*d2[c])

    X2 = sparse.coo_matrix((X2_data, (X.row, X.col)), dtype=np.float64)
    return X2


@cython.boundscheck(False)
@cython.wraparound(False)
cdef C_get_s(X, int[::1] p, int[::1] q, int c_true, double[::1] pdp, double[::1] qdq, double eps, double[::1] s):
    cdef int i = 0
    for i in range(c_true):
        s[i] = 0

    cdef int r = 0
    cdef int c = 0
    cdef double v = 0
    cdef int c1 = 0
    cdef int c2 = 0

    cdef int[::1] row_view = X.row
    cdef int[::1] col_view = X.col
    cdef double[::1] data_view = X.data

    for i in range(X.nnz):
        r = row_view[i]
        c = col_view[i]
        v = data_view[i]
        c1 = p[r]
        c2 = q[c]
        if c1 == c2:
            s[c1] += v

    for i in range(c_true):
        s[i] /= pdp[i]*qdq[i] + eps

# sum v by y, ret[c] = sum v[i], y[i] = c
@cython.boundscheck(False)
@cython.wraparound(False)
def C_sum_vy(double[::1] v, int[::1] y, int c, double[::1] ret):

    cdef int N = v.shape[0]
    cdef int tmp_c = 0

    cdef int i = 0

    for i in range(c):
        ret[i] = 0

    for i in range(N):
        tmp_c = y[i]
        ret[tmp_c] += v[i]


@cython.boundscheck(False)
@cython.wraparound(False)
# A: sparse, B: dense
cdef C_EuDist2(A, double[::1] va_, int va_flag,
              B, double[::1] vb_, int vb_flag,
              int row, C):

    if va_flag==0:
        C_L2norm_sparseA(A, row, 1, va_)

    if vb_flag==0:
        C_L2norm_denseA(B, row, 1, vb_)

    if row==1:
        # dot(A, 0, B, 1, AB_)
        tmp = A.dot(B.T)
    else:
        # dot(A, 1, B, 0, AB_)
        tmp = A.T.dot(B)

    tmp = np.ascontiguousarray(tmp, dtype=np.float64)
    C_EuDist2_sub(va_, vb_, tmp, C)



cdef C_EuDist2_sub(double[::1] va_, double[::1] vb_, double[:, ::1] AB_, double[:, ::1] C):
    cdef int n = C.shape[0]
    cdef int m = C.shape[1]

    cdef int i = 0
    cdef int j = 0
    for i in prange(n, nogil=True):
        for j in range(m):
            C[i, j] = va_[i] + vb_[j] - 2*AB_[i, j]


@cython.boundscheck(False)
@cython.wraparound(False)
# A: dense_matrix nxd
cdef C_L2norm_denseA(A, int row, int squared, double[::1] norm):
    cdef int n = A.shape[0]
    cdef int d = A.shape[1]
    cdef int i = 0

    cdef double v = 0

    if row==1:
        for i in range(n):
            norm[i] = 0
            for j in range(d):
                v = A[i, j]
                norm[i] += v * v
    else:
        for i in range(d):
            norm[i] = 0
            for j in range(n):
                v = A[j, i]
                norm[i] += v * v


@cython.boundscheck(False)
@cython.wraparound(False)
# A: coo_matrix
cdef C_L2norm_sparseA(A, int row, int squared, double[::1] norm):
    A = sparse.coo_matrix(A)

    cdef int n = 0
    cdef int i = 0
    cdef int r = 0
    cdef double v = 0
    cdef int[::1] ind

    if row==1:
        n = A.shape[0]
        ind = A.row
    else:
        n = A.shape[1]
        ind = A.col

    # init
    for i in range(n):
        norm[i] = 0

    cdef double[::1] data_view = A.data
    for i in range(A.nnz):
        r = ind[i]
        v = data_view[i]
        norm[r] += v*v

    if squared==0:
        for i in range(n):
            norm[i] = np.sqrt(norm[i])


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def A_norm(np.ndarray[double, ndim=2, mode="c"] A,
#             np.ndarray[double, ndim=1, mode="c"] va):
#
#     cdef int n = A.shape[0]
#     cdef int d = A.shape[1]
#
#     for i in range(n):
#         va[i] = 0
#
#     for i in range(n):
#         for j in range(d):
#                 va[i] += A[i, j]*A[i, j]

# sum M by p and q, ret[c1, c2] = sum M[i, j], p[i] = c1, q[j] = c2
# ret = P.T M Q
# def sum_Mpq(np.ndarray[double, ndim=2, mode="c"] M,
#            np.ndarray[int, ndim=1, mode="c"] p,
#            np.ndarray[int, ndim=1, mode="c"] q,
#            int c,
#            np.ndarray[double, ndim=2, mode="c"] ret):
#
#     cdef int n = p.shape[0]
#     cdef int m = q.shape[0]
#
#     cdef int c1 = 0
#     cdef int c2 = 0
#
#     for i in range(c):
#         for j in range(c):
#             ret[i, j] = 0
#
#     for i in range(n):
#         for j in range(m):
#             c1 = p[i]
#             c2 = q[j]
#             ret[c1, c2] += M[i, j]


# C = Eudist2(A, B)
# va[i] = ||Ai||_2^2
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def EuDist2(np.ndarray[double, ndim=2, mode="c"] A,
#             np.ndarray[double, ndim=1, mode="c"] va_,
#             np.ndarray[double, ndim=2, mode="c"] B,
#             np.ndarray[double, ndim=1, mode="c"] vb_,
#             np.ndarray[double, ndim=2, mode="c"] ABT_,
#             np.ndarray[double, ndim=2, mode="c"] C):
#
#     A_norm(A, va_)
#     A_norm(B, vb_)
#     ABT_ = A.dot(B.T)
#
#     cdef int n = A.shape[0]
#     cdef int m = B.shape[0]
#
#     cdef int i = 0
#     cdef int j = 0
#     for i in prange(n, nogil=True):
#         for j in range(m):
#             C[i, j] = va_[i] + vb_[j] - 2*ABT_[i, j]



# sparse A, dense B
# cdef dot(A, int transA, B, int transB, C):
#     cdef int ar = A.shape[0]
#     cdef int ac = A.shape[1]
#     cdef int br = B.shape[0]
#     cdef int bc = B.shape[1]
#
#     cdef int i = 0
#     cdef int j = 0
#     cdef int r = 0
#     cdef int c = 0
#     cdef double v = 0
#
#     cdef int nnz = A.nnz
#
#     cdef int[::1] row_view = A.row
#     cdef int[::1] col_view = A.col
#     cdef double[::1] data_view = A.data
#
#     if transA == 0 and transB == 1:
#         # C = A * BT
#         C = np.zeros((ar, br))
#         for i in range(nnz):
#             r = row_view[i]
#             c = col_view[i]
#             v = data_view[i]
#             for j in range(br):
#                 C[r, j] += v*B[j, c]
#
#     if transA == 1 and transB == 0:
#         # C = AT * B
#         C = np.zeros((ac, bc))
#         for i in range(A.nnz):
#             r = A.row[i]
#             c = A.col[i]
#             v = A.data[i]
#             C[c, :] += v*B[r, :]
