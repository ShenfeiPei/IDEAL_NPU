# abandon
import numpy as np
import IDEAL_NPU.funs as Ifuns
from backup import ISR

data_name = "srbct"
X, y_true, N, dim, c_true = Ifuns.load_mat("D:/DATA/" + data_name)

print(data_name, N, dim, c_true)

obj = ISR(X, c_true)

knn_list = [10, 20, 30, 40, 50]
acc = np.zeros(len(knn_list))
for i, knn in enumerate(knn_list):
    y = obj.clu(graph_knn=knn, sr_ITER=10)
    acc[i] = Ifuns.accuracy(y_true, y)

print(acc)
print(np.mean(acc))

# paper: USPS, ACC, 67.52 %
