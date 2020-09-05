from libcpp.vector cimport vector

from Keep_order cimport Keep_order

cdef extern from "my.cpp":
    pass

cdef extern from "my.h":
    void argsort_f(int *v, int n, int *ind)
    void symmetry(vector[vector[int]] &NN, vector[vector[double]] &NND, double fill_ele);

cdef extern from "EEBC.cpp":
    pass

cdef extern from "EEBC.h":
    cdef cppclass EEBC:
        int N
        int knn
        int c_true
        vector[vector[int]] NN
        vector[vector[double]] NND
        vector[int] y

        int *hi
        int *hi_TF
        int *hi_count

        int *knn_c
        double max_d

        Keep_order KO

        EEBC() except +
        EEBC(vector[vector[int]]& NN, vector[vector[double]]& NND, int c_true) except +
        void opt()
        void construct_hi(int sam_i)
