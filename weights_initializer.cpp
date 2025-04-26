#include <random>
#include <cmath>
#include "weights_initializer.h"

namespace galanet::weight_initializers {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    Matrix zeros(int num_rows, int num_cols) {
        return Matrix(num_rows, num_cols, 0);
    }

    Matrix ones(int num_rows, int num_cols) {
        return Matrix(num_rows, num_cols, 1);
    }

    Matrix random_uniform(int num_rows, int num_cols) {
        return random_uniform(num_rows, num_cols, -1.0, 1.0);
    }

    Matrix random_uniform(int num_rows, int num_cols, double min_val, double max_val) {
        std::uniform_real_distribution<double> dist(min_val, max_val);
        Matrix result(num_rows, num_cols);
        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < num_cols; j++) {
                result(i, j) = dist(gen);
            }
        }
        return result;
    }

    Matrix he_uniform(int num_rows, int num_cols) {
        double limit = std::sqrt(2.0 / num_rows);
        return random_uniform(num_rows, num_cols, -limit, limit);
    }

    Matrix xavier_uniform(int num_rows, int num_cols) {
        double limit = std::sqrt(2.0 / (num_rows + num_cols));
        return random_uniform(num_rows, num_cols, -limit, limit);
    }
}