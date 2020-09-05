#include "EEBC.h"

EEBC::EEBC(){}

EEBC::EEBC(std::vector<std::vector<int>> &NN, std::vector<std::vector<double>> &NND, int c_true){
    this->N = NN.size();
    this->knn = NN[0].size();
    this->c_true = c_true;
    this->NN = NN;
    this->NND = NND;

    // allocate memory
    y.assign(N, 0);

    hi = new int[c_true];
    hi_TF = new int[c_true];
    hi_count = new int[c_true];
    knn_c = new int[knn];

    // initialize
    srand((unsigned)time(NULL));
    std::generate(y.begin(), y.end(), [c_true]() {return rand() % c_true;});
//    init_randi(0, c_true, y, N);
    KO = Keep_order(y, N, c_true);

//    show_M_int(NN, N, knn, 2, 5);
//    show_M_dou(NND, N, knn, 2, 5);
    max_d = maximum_2Dvec(NND);
    std::cout << "max_d = " << max_d << std::endl;

    symmetry(NN, NND, max_d);
}

EEBC::~EEBC() {}

void EEBC::opt(){
    clock_t t_start = clock();

    int end_count = 0;
    int iter = 0, c_old = 0, c_new = 0;

    double h_min_val = 0;
    int h_min_ind = 0;

    int tmp_c = 0, tmp_nb = 0;

    for (iter = 0; iter < 1000; iter++){
        for (int sam_i = 0; sam_i < N; sam_i++){
            c_old = y[sam_i];

            // knn_c
            for (int k = 0; k < knn; k++){
                tmp_nb = NN[sam_i][k];
                knn_c[k] = y[tmp_nb];
            }

            construct_hi(sam_i);

            // find minimun in knn
            h_min_ind = knn_c[0];
            h_min_val = hi[h_min_ind];
            for (int k = 1; k < knn; k++){
                tmp_c = knn_c[k];
                if (hi[tmp_c] < h_min_val){
                    h_min_ind = tmp_c;
                    h_min_val = hi[h_min_ind];
                }
            }

            if (KO.o2ni[0] * max_d < h_min_val){
                c_new = KO.o2c[0];
            }else{
                c_new = h_min_ind;
            }

            if (c_new == c_old){
                end_count +=1;
            }else{
                y[sam_i] = c_new;
                end_count = 0;

                KO.sub(KO.c2o[c_old]);
                KO.add(KO.c2o[c_new]);
            }
        }

        if (end_count > 2*N){
            break;
        }
    }
    std::cout << "Iter = " << iter << std::endl;
    clock_t t_end = clock();
    std::cout << "time = " << t_end-t_start << std::endl;
}


void EEBC::construct_hi(int sam_i){

    int tmp_c, tmp_ni;

    for (int k = 0; k < knn; k++){
        tmp_c = knn_c[k];
        hi[tmp_c] = 0;
        hi_count[tmp_c] = 0;
        hi_TF[tmp_c] = 0;
    }

    for (int k = 0; k < knn; k++){
        tmp_c = knn_c[k];
        hi[tmp_c] += NND[sam_i][k];
        hi_count[tmp_c] += 1;
    }

    for (int j = 0; j < knn; j++){
        tmp_c = knn_c[j];
        if (hi_TF[tmp_c] == 0){
            hi_TF[tmp_c] = 1;

            tmp_ni = KO.o2ni[KO.c2o[tmp_c]];
            hi[tmp_c] += (tmp_ni - hi_count[tmp_c]) * max_d;
        }
    }

}
