import time
import numpy as np
import IDEAL_NPU.funs as Ifuns
from IDEAL_NPU.cluster import DNC

data_name = "nci"
X, y_true, N, dim, c_true = Ifuns.load_mat("D:/DATA/" + data_name)
print(data_name, N, dim, c_true)

knn = 10
obj = DNC(X, c_true)
y = obj.clu(graph_knn=knn, ITER=100)

acc = Ifuns.accuracy(y_true, y)
print(acc)

# paper: DNC, nci, ACC, 74.1 %
