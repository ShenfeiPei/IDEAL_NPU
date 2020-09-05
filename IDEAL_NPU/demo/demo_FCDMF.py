import numpy as np
from IDEAL_NPU.cluster import FCDMF
from IDEAL_NPU import funs as Ifuns

X, y_true, N, dim, c_true = Ifuns.load_mat("D:/DATA/face94_256.mat")
X = Ifuns.normalize_fea(X, 0)
print(N, dim, c_true)

obj = FCDMF(X, c_true)
Y = obj.clu(KTIMES=5)

ret = [Ifuns.accuracy(y_true, y) for y in Y]
print(ret)
print(np.mean(ret))

