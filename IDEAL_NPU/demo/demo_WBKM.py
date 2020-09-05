import numpy as np
from IDEAL_NPU.cluster import WBKM
from IDEAL_NPU import funs as Ifuns

import scipy.io as sio

data = sio.loadmat("D:/DATA/TDT2.mat")
X = data["fea"]
y_true = data["gnd"].reshape(-1)
c_true = len(np.unique(y_true))
print(X.shape, c_true)

# X, y_true, N, dim, c_true = Ifuns.load_mat("D:/DATA/TDT2.mat")
# print(N, dim, c_true)

obj = WBKM(X, c_true)
y = obj.clu(ITER=100)

ret = Ifuns.precision(y_true, y)
print(ret)

ret = Ifuns.ami(y_true, y)
print(ret)

ret = Ifuns.ari(y_true, y)
print(ret)


# ret = [Ifuns.accuracy(y_true, y) for y in Y]
# print(ret)
# print(np.mean(ret))

