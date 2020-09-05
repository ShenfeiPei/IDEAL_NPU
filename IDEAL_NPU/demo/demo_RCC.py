import numpy as np
import IDEAL_NPU.funs as Ifuns
from IDEAL_NPU.cluster import RCC

data_name = "pendigits"
X, y_true, N, dim, c_true = Ifuns.load_mat("D:/DATA/" + data_name)
print(data_name, N, dim, c_true)

knn = 10
thrs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
y = RCC(k=knn, clustering_threshold=thrs[2], verbose=False).fit(X)

acc = Ifuns.ami(y_true, y, average_method="max")
print(acc)

