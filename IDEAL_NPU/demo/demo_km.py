import numpy as np
import IDEAL_NPU.funs as Ifuns


def main():
    data_name = "CALFW_29v2"
    data_path = "D:/DATA/"
    if "29v2" in data_name:
        X, y_true, N, dim, c_true = Ifuns.load_mat(data_path + "FaceData/" + data_name)
    else:
        X, y_true, N, dim, c_true = Ifuns.load_mat(data_path + data_name)

    print("hh")
    Y = Ifuns.kmeans(X, c_true, rep=10, init="random", par=1)
    acc = np.array([Ifuns.accuracy(y_true, y) for y in Y])
    ari = np.array([Ifuns.ari(y_true, y) for y in Y])
    ami = np.array([Ifuns.ami(y_true, y) for y in Y])
    nmi = np.array([Ifuns.nmi(y_true, y) for y in Y])
    print("{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(data_name,
                                                                                      np.mean(acc), np.std(acc),
                                                                                      np.mean(ari), np.std(ari),
                                                                                      np.mean(ami), np.std(ami),
                                                                                      np.mean(nmi), np.std(nmi)))


if __name__ == '__main__':
    main()
