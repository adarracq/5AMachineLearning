#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;

extern "C" {

    DLLEXPORT int my_add(int x, int y) {
        return x + y;
    }

    DLLEXPORT int my_mul(int x, int y) {
        return x * y;
    }

    DLLEXPORT double* linear_model_create(int dimension) {
        int dim = dimension + 1;
        auto tab = new double[dim];
        for (int i = 0; i < dim; i++) {

            tab[i] = ((double)rand())/ RAND_MAX;
            //tab[i] = tab[i] / 1000;
        }
        return tab;
    }

    DLLEXPORT double linear_model_predict_regression(double* model,
        double* tabInput, int dataset_inputs_ligne_count) {
        double sum = model[0];
        for (int i = 1; i < dataset_inputs_ligne_count+1; i++) {
            sum += model[i] * tabInput[i - 1];
        }
        return sum;
    }

    DLLEXPORT double linear_model_predict_classification(double* model, double* tabModelInput, int dataset_inputs_ligne_count) {
        if (linear_model_predict_regression(model, tabModelInput, dataset_inputs_ligne_count) >= 0) {
            return 1;
        }
        else {
            return -1;
        }
    }

    DLLEXPORT void linear_model_train_classification(double* model,
        double* dataset_inputs,
        double* dataset_expected_outputs,
        int dataset_inputs_ligne_count,
        int dataset_inputs_ligne_size,
        int iterations_count,
        double alpha) {


        for (int it = 0; it < iterations_count; it++) {
            //int k = rand() % dataset_inputs_ligne_count;
            for (int k = 0; k < dataset_inputs_ligne_count; k++) {
                auto tab_temp = new double[dataset_inputs_ligne_size];
                for (int a = 0; a < dataset_inputs_ligne_size; a++) {
                    tab_temp[a] = dataset_inputs[(k * dataset_inputs_ligne_size) + a];
                }
                double g_x_k = linear_model_predict_classification(model, tab_temp, dataset_inputs_ligne_size);
                double grad = alpha * (dataset_expected_outputs[k] - g_x_k);
                model[0] += grad * 1;
                for (int i = 0; i < dataset_inputs_ligne_size; i++) {
                    model[i + 1] += grad * dataset_inputs[(k * dataset_inputs_ligne_size) + i];
                }
                delete[] tab_temp;
            }

        }
    }

    DLLEXPORT void linear_model_train_regression(double* model,
        double* dataset_inputs,
        double* dataset_expected_outputs,
        int dataset_inputs_ligne_count,
        int dataset_inputs_ligne_size) {


        Eigen::MatrixXd X(dataset_inputs_ligne_count, dataset_inputs_ligne_size + 1);
        Eigen::MatrixXd Y(dataset_inputs_ligne_count, 1);

        for (int i = 0; i < dataset_inputs_ligne_count; ++i) {
            X(i, 0) = 1;
            Y(i, 0) = dataset_expected_outputs[i];
            for (int j = 1; j < (dataset_inputs_ligne_size + 1); j++) {
                X(i, j) = dataset_inputs[i * dataset_inputs_ligne_size + (j - 1)];
            }
        }

        Eigen::MatrixXd W = ((X.transpose() * X).inverse() * X.transpose()) * Y;

        for (int i = 0; i < dataset_inputs_ligne_size + 1; ++i) {
            model[i] = W(i, 0);
        }
    }
    /*
    struct MLP {
        int* npl;
        int npl_size;
        double*** w;
        double** x;
        double** deltas;
    };

    DLLEXPORT struct MLP* mlp_model_create(int* npl, int npl_size) {
        MLP mlp = new MLP();

    }








    DLLEXPORT void Perceptron_multicouches(double* model,
        MLP perceptron,
        double* dataset_expected_outputs,
        int dataset_inputs_ligne_count,
        int dataset_inputs_ligne_size) {

    }*/


}