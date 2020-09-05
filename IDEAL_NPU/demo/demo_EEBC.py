import numpy as np
import IDEAL_NPU.funs as Ifuns
from IDEAL_NPU.cluster import EEBC, EEBCX


data_name = "mpeg7"
X, y_true, N, dim, c_true = Ifuns.load_mat("D:/DATA/"+data_name)
print(N, dim, c_true)

# EEBC
knn = 20
D = Ifuns.EuDist2(X, X, squared=True)
ind_M = np.argsort(D, axis=1)
NN = ind_M[:, :knn]
NND = Ifuns.matrix_index_take(D, NN)

obj = EEBC(NN.astype(np.int32), NND, c_true)
obj.clu()
y_pred = obj.y_pre

acc = Ifuns.accuracy(y_true=y_true, y_pred=y_pred)
print(acc)

# EEBC-X
obj = EEBCX(X, c_true)
obj.clu()

y_pred = obj.y
acc = Ifuns.accuracy(y_true=y_true, y_pred=y_pred)
print(acc)
