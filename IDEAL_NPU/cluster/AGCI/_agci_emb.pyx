cimport numpy as np
import numpy as np
np.import_array()

def emb(np.ndarray[double, ndim=2, mode="c"] Z, int c, int ITER):
    cdef double lam = 0.5
    cdef np.ndarray[double, ndim=2, mode="c"] Delta_inv = np.diag(1/np.sum(Z, axis=0))
    cdef np.ndarray[double, ndim=2, mode="c"] F = np.random.rand(Z.shape[0], c)

    cdef np.ndarray[double, ndim=2, mode="c"] P
    cdef np.ndarray[double, ndim=2, mode="c"] Q

    for Iter in range(ITER):

        P = 2*lam*F + Z@(Delta_inv@(Z.T@F))
        Q = F + 2*lam*F@(F.T@F)
        F = F*np.sqrt(P/Q)

    return F

