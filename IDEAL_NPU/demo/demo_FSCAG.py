import numpy as np
from IDEAL_NPU.cluster import FSCAG
from IDEAL_NPU import funs as Ifuns

X, y_true, N, dim, c_true = Ifuns.load_mat("D:/DATA/face94_256.mat")
X = Ifuns.normalize_fea(X, 0)
print(N, dim, c_true)

obj = FSCAG(X, c_true)
y = obj.clu()

ret = Ifuns.accuracy(y_true, y)
print(ret)

