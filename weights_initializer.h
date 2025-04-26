#pragma once
#include "matrix.h"

namespace galanet::weight_initializers {
    Matrix zeros(int num_rows, int num_cols);
    Matrix ones(int num_rows, int num_cols);
    Matrix random_uniform(int num_rows, int num_cols);  // Base version
    Matrix random_uniform(int num_rows, int num_cols, double min_val, double max_val);  // Extended version
    Matrix xavier_uniform(int num_rows, int num_cols);
    Matrix he_uniform(int num_rows, int num_cols);
}