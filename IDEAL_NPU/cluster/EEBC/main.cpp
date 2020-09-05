#include <iostream>
#include "EEBC.h"
#include "my.h"
using namespace std;

int main()
{
    srand((unsigned)time(NULL));

    int N = 1400;
    int K = 20;
    int knn = 20;
    int c_true = 70;

    vector<vector<int>> NN(N, vector<int> (knn, 0));
    vector<vector<double>> NND(N, vector<double>(knn, 0));

    vector<int> y_true(N, 0);

    string data_name = "mpeg7";
    string y_pred_name = "D:/project/EEBC/data/" + data_name +".y_pred";
    string NNname = "D:/project/EEBC/data/" + data_name + ".graph";
    string NNDname = "D:/project/EEBC/data/" + data_name + ".e2";

    read_M_vec2<int>(NNname, K, NN);
    read_M_vec2<double>(NNDname, K, NND);

    show_2Dvec<int>(NN, 5, 5);
    show_2Dvec<double>(NND, 5, 5);
    cout << "reading END" << endl;

    EEBC obj = EEBC(NN, NND, c_true);

    obj.opt();

    write_vec(y_pred_name, obj.y);
//    float acc = clu_acc(y_true, obj.y, N);
//    cout << acc << endl;
    return 0;
}
