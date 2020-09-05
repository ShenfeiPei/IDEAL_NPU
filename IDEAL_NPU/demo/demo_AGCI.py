import numpy as np
import IDEAL_NPU.funs as Ifuns
from IDEAL_NPU.cluster import AGCI


X, y_true, N, dim, c_true = Ifuns.load_Agg()

obj = AGCI(X, c_true)
y_pred = obj.clu(graph_knn=20, km_times=10)

pre = Ifuns.precision(y_true=y_true, y_pred=y_pred)
rec = Ifuns.recall(y_true=y_true, y_pred=y_pred)
f1 = 2 * pre * rec / (pre + rec)

print("{:.3f}".format(pre))
print("{:.3f}".format(f1))

# not verified

