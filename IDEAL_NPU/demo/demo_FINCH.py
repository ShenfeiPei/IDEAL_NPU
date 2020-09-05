import numpy as np
import IDEAL_NPU.funs as Ifuns
from IDEAL_NPU.cluster import FINCH

X, y_true, N, dim, c_true = Ifuns.load_Agg()
print("Agg", N, dim, c_true)

# FINCH (1)
obj = FINCH(data=X, req_clust=c_true, distance='euclidean')
Y, num_clu, req_y = obj.clu()

acc = Ifuns.nmi(y_true, req_y)
print(acc)

# FINCH (2)
# Y, num_clu, req_y = FINCH(X, distance='euclidean')  # or cosine
#
# acc = [Ifuns.nmi(y_true, y) for y in Y]
# print(acc)
