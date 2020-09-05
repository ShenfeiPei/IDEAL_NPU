import numpy as np
import IDEAL_NPU.funs as Ifuns
from IDEAL_NPU.cluster import SC

X, y_true, N, dim, c_true = Ifuns.load_USPS()
print("USPS", N, dim, c_true)

knn = 20
obj = SC(X, c_true)
y = obj.clu(graph_knn=knn, km_rep=10, km_init="random")

acc = Ifuns.accuracy(y_true, y)
ari = Ifuns.ari(y_true, y)
ami = Ifuns.ami(y_true, y)
nmi = Ifuns.nmi(y_true, y)
print("{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format("USPS",
                                                                                  np.mean(acc), np.std(acc),
                                                                                  np.mean(ari), np.std(ari),
                                                                                  np.mean(ami), np.std(ami),
                                                                                  np.mean(nmi), np.std(nmi)))
