import numpy as np
import IDEAL_NPU.funs as Ifuns
from IDEAL_NPU.cluster import SNN
from IDEAL_NPU.cluster import SNN_opt

X, y_true, N, dim, c_true = Ifuns.load_Agg()
print("Agg", N, dim, c_true)

option = SNN_opt()
option.shouldUseOrdinalAssignment = False

knn = 15
centroid, y = SNN(knn, N, dim, c_true).Run(X)

ami = Ifuns.ami(y_true, y)
print(ami)
ari = Ifuns.ari(y_true, y)
print(ari)

# paper: Knn=15, Agg, AMI=0.9500, ARI=0.9594
