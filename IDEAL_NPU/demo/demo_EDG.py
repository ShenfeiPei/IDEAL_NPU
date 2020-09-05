import numpy as np
from sklearn.metrics.pairwise import euclidean_distances as EuDist2

import IDEAL_NPU.funs as Ifuns
from IDEAL_NPU.cluster import EDG


X, y_true, N, dim, c_true = Ifuns.load_Agg()
D_full = EuDist2(X, X, squared=True)
NN_full = np.argsort(D_full, axis=1)

knn_list = [5, 6, 7, 8, 9, 10]
Y = np.zeros((len(knn_list), N))
for i, knn in enumerate(knn_list):
    NN = NN_full[:, 1:(knn+1)]
    NND = Ifuns.matrix_index_take(D_full, NN)

    EDG_obj = EDG(NN, NND)
    y = EDG_obj.cluster()
    Y[i, :] = y

pre = np.array([Ifuns.precision(y_true=y_true, y_pred=y_pred) for y_pred in Y])
rec = np.array([Ifuns.recall(y_true=y_true, y_pred=y_pred) for y_pred in Y])
f1 = 2 * pre * rec / (pre + rec)
ind = np.argmax(f1)

print("{}".format(pre[ind]))
print("{}".format(f1[ind]))
# print("{}".format(fmi))

