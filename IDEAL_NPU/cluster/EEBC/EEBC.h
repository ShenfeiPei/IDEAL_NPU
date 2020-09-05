#ifndef _EEBC_H
#define _EEBC_H

#include <iostream>
#include <algorithm>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <ctime>
#include "my.h"
#include "Keep_order.h"

class EEBC{
public:
    int N = 0;
    int knn = 0;
    int c_true = 0;

    std::vector<std::vector<int>> NN;
    std::vector<std::vector<double>> NND;
    std::vector<int> y;

    int *hi = nullptr;
    int *hi_TF = nullptr;
    int *hi_count = nullptr;

    int *knn_c = nullptr;

    double max_d = 0;

    Keep_order KO;

    EEBC();
    EEBC(std::vector<std::vector<int>> &NN, std::vector<std::vector<double>> &NND, int c_true);
    ~EEBC ();

    void opt();

    void construct_hi(int sam_i);

};
#endif
