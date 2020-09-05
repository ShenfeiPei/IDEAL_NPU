cimport numpy as np
import numpy as np
np.import_array()

def get_M(
        np.ndarray[int, ndim=1, mode="c"] y,
        int c,
        np.ndarray[double, ndim=1, mode="c"] Da):

    cdef int N = len(Da)

    cdef np.ndarray[double, ndim=1, mode="c"] ydy_sq = np.zeros(c)
    for i in range(N):
        ydy_sq[y[i]] += Da[i]

    for i in range(c):
        ydy_sq[i] = np.sqrt(ydy_sq[i] + 2.2204e-16)

    cdef np.ndarray[double, ndim=2, mode="c"] M = np.zeros((c, N))
    cdef int tmp_c = 0
    for j in range(N):
        tmp_c = y[j]
        M[tmp_c, j] = 1/ydy_sq[tmp_c]*np.sqrt(Da[j])

    return M

def update_y(
        np.ndarray[double, ndim=2, mode="c"] G,
        np.ndarray[int, ndim=1, mode="c"] y,
        int c,
        np.ndarray[double, ndim=1, mode="c"] Da):

    cdef int N = len(Da)

    cdef np.ndarray[double, ndim=1, mode="c"] s1 = np.zeros(c)
    cdef np.ndarray[double, ndim=1, mode="c"] s2 = np.zeros(c)
    cdef np.ndarray[double, ndim=1, mode="c"] s3 = np.zeros(c)
    cdef np.ndarray[double, ndim=1, mode="c"] s4 = np.zeros(c)
    cdef np.ndarray[double, ndim=1, mode="c"] s5 = np.zeros(c)

    cdef np.ndarray[double, ndim=1, mode="c"] ydy = np.zeros(c)
    for i in range(N):
        ydy[y[i]] += Da[i]

    cdef int tmp_c = 0
    cdef int cou = 0
    cdef int flag = 0
    cdef int ITER = 100

    for iter in range(ITER):
        for i in range(N):
            c_old = y[i]

            # S = (s1 + s2)/s3 - (s1 - s4)/s5

            # s1
            for j in range(c):
                s1[j] = 0
            for t in range(N):
                tmp_c = y[t]
                s1[tmp_c] += np.sqrt(Da[t])*G[t, tmp_c]

            # s2
            for j in range(c):
                s2[j] = np.sqrt(Da[i])*G[i, j]
            s2[y[i]] = 0

            # s3
            for j in range(c):
                s3[j] = np.sqrt(ydy[j] + Da[i])
            s3[y[i]] = np.sqrt(ydy[y[i]])

            # s4
            for j in range(c):
                s4[j] = 0
            s4[y[i]] = np.sqrt(Da[i])

            # s5
            for j in range(c):
                s5[j] = np.sqrt(ydy[j]) + 2.2204e-16
            s5[y[i]] = np.sqrt(ydy[y[i]] - Da[i]) + 2.2204e-16

            si = (s1+s2)/s3 - (s1 - s4)/s5
            c_new = np.argmax(si)

            y[i] = c_new
            ydy[c_old] -= Da[i]
            ydy[c_new] += Da[i]

            if c_new == c_old:
                cou += 1
                if cou > 2*N:
                    flag = 1
                    break
            else:
                cou = 0

        if flag==1:
            break

    return y







