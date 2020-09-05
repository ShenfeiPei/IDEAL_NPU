cimport numpy as np
import numpy as np
np.import_array()

class EEBCX(object):
    def __init__(self, X, c_true):
        self.X = X
        self.N = X.shape[0]
        self.dim = X.shape[1]
        self.c_true = c_true
        self.y = np.random.randint(0, self.c_true, self.N, dtype=np.int32)

    def clu(self, gamma = 0, ITER=100):
        cdef np.ndarray[double, ndim=1] norm = np.sum(self.X*self.X, axis=1)
        cdef np.ndarray[double, ndim=1] norm_sum = np.zeros(self.c_true)
        cdef np.ndarray[double, ndim=2] X_sum = np.zeros((self.dim, self.c_true))
        cdef np.ndarray[int, ndim=1] n_sum = np.zeros(self.c_true, dtype=np.int32)

        cdef int i = 0
        cdef tmp_c = 0

        #    n_sum[c] = sum [xi in c]    1          : c x 1
        #    X_sum[c] = sum [xi in c]    xi         : d x c
        # norm_sum[c] = sum [xi in c] || xi ||_2^2  : c x 1
        for i in range(self.N):
            tmp_c = self.y[i]
            n_sum[tmp_c] += 1
            X_sum[:, tmp_c] += self.X[i, :]
            norm_sum[tmp_c] += norm[i]

        cdef int Iter = 0
        cdef int flag = 0
        cdef int c_old = 0
        cdef int c_new = 0

        cdef np.ndarray[double, ndim=1] sub_problem = np.zeros(self.c_true)

        for Iter in range(ITER):
            flag = 0
            for i in range(self.N):
                c_old = self.y[i]
                sub_problem = (gamma + norm[i]) * n_sum + norm_sum - 2*(self.X[i, :].reshape(1, -1)@X_sum).reshape(-1)
                sub_problem[c_old] -= gamma
                c_new = np.argmin(sub_problem)

                if c_old != c_new:
                    flag = 1
                    self.y[i] = c_new
                    norm_sum[c_old] -= norm[i]
                    norm_sum[c_new] += norm[i]
                    X_sum[:, c_old] -= self.X[i, :]
                    X_sum[:, c_new] += self.X[i, :]
                    n_sum[c_old] -= 1
                    n_sum[c_new] += 1

            if flag == 0:
                break

    @property
    def y_pre(self):
        return self.y

