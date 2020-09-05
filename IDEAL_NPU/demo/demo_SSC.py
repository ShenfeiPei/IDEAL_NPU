import numpy as np
import IDEAL_NPU.funs as Ifuns
from IDEAL_NPU.cluster import SSC


X, y_true, N, dim, c_true = Ifuns.load_USPS()
print("USPS", N, dim, c_true)

obj = SSC(X=X, c_true=c_true)
y = obj.clu(km_rep=10, way="NJW")

acc = Ifuns.accuracy(y_true, y)
print(acc)

# paper: USPS, ACC, 67.52 %
