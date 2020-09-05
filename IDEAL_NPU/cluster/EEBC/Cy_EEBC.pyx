# distutils: language = c++

cimport numpy as np
import numpy as np
np.import_array()

from EEBC cimport EEBC

cdef class PyEEBC:
    cdef EEBC c_EEBC

    def __cinit__(self, np.ndarray[int, ndim=2] NN, np.ndarray[double, ndim=2] NND, c_true):
        self.c_EEBC = EEBC(NN, NND, c_true)

    def clu(self):
        self.c_EEBC.opt()

    def _construct_hi(self, sam_i):
        self.c_EEBC.construct_hi(sam_i)

    @property
    def y_pre(self):
        return self.c_EEBC.y
