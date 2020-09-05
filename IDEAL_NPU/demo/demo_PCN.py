import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as EuDist2

import IDEAL_NPU.funs as Funs
from IDEAL_NPU.cluster import PCN


X, y_true, N, dim, c_true = Funs.load_Agg()
D_full = EuDist2(X, X, squared=True)
NN_full = np.argsort(D_full, axis=1)

knn = 33
NN = NN_full[:, 1:(knn+1)]
NND = Funs.matrix_index_take(D_full, NN)

for i in range(N):
    tmp_ind = np.lexsort((NN[i, :], NND[i, :]))
    NN[i, :] = NN[i, tmp_ind]

print("begin")
PCN_obj = PCN(NN, NND)
y_pred = PCN_obj.cluster()
t = PCN_obj.get_time()

print("end", t)
pre = Funs.precision(y_true=y_true, y_pred=y_pred)
rec = Funs.recall(y_true=y_true, y_pred=y_pred)
f1 = 2 * pre * rec / (pre + rec)

print("{}".format(pre))
print("{}".format(f1))

# print("{}".format(pre[ind]))
# print("{}".format(f1[ind]))
# print("{}".format(fmi))
