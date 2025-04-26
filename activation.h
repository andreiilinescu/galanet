#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "matrix.h"

namespace galanet::activation {
    Matrix tanh(const Matrix &input);
    Matrix tanhDerivative(const Matrix &input);

    double relu(double x);
    double reluDerivative(double x);
    Matrix relu(const Matrix &input);
    Matrix reluDerivative(const Matrix &input);

    Matrix softmax(const Matrix &input);
    Matrix softmaxDerivative(const Matrix &softmaxOutput);
}

#endif