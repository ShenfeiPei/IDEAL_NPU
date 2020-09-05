import ctypes
import os
import numpy as np


class EDG(object):
    def __init__(self, NN, NND):
        self.N, self.knn = NN.shape

        this_directory = os.path.abspath(os.path.dirname(__file__))
        self.lib = np.ctypeslib.load_library('_edg.dll', this_directory)

        self.lib.clu_new.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int32, ctypes.c_int32]
        self.lib.clu_new.restype = ctypes.c_void_p

        self.obj = self.lib.clu_new(NN.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                               NND.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.N, self.knn)

    def cluster(self):
        self.lib.clustering.argtypes = [ctypes.c_void_p]
        self.lib.clustering.restype = ctypes.POINTER(ctypes.c_int32)
        y_pre = self.lib.clustering(self.obj)
        y = np.ctypeslib.as_array(y_pre, (self.N,))
        return y

    def get_time(self):
        self.lib.get_time.argtypes = [ctypes.c_void_p]
        self.lib.get_time.restype = ctypes.c_double
        return self.lib.get_time(self.obj)

    def get_den(self):
        self.lib.get_den.argtypes = [ctypes.c_void_p]
        self.lib.get_den.restype = ctypes.POINTER(ctypes.c_int32)
        den = self.lib.get_den(self.obj)
        return np.ctypeslib.as_array(den, (self.N,))

    def get_row_col(self):
        self.lib.get_row_col.argtypes = [ctypes.c_void_p]
        self.lib.get_row_col.restype = ctypes.POINTER(ctypes.c_int32)
        row_col = self.lib.get_row_col(self.obj)
        nlink = row_col[0]
        rc = np.ctypeslib.as_array(row_col, (2*nlink + 1,))
        r = rc[1:(nlink+1)]
        c = rc[(nlink+1):]
        return r, c

    def get_NNS(self):
        self.lib.get_NNS.argtypes = [ctypes.c_void_p]
        self.lib.get_NNS.restype = ctypes.POINTER(ctypes.c_double)
        C_NNS = self.lib.get_NNS(self.obj)
        NNS = np.ctypeslib.as_array(C_NNS, (self.N, self.knn))
        return NNS

    @staticmethod
    def add(i, j):
        return i+j

