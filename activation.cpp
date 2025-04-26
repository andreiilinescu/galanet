#include <cmath>
#include "activation.h"
#include <iostream>
namespace galanet::activation {

    Matrix tanh(const Matrix &m) {
        Matrix res(m.getRows(), m.getCols());
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                res(i,j) = std::tanh(m(i,j));
            }
        }
        return res;
    }

    Matrix tanhDerivative(const Matrix &m) {
        Matrix res(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                double t = std::tanh(m(i, j));
                res(i, j) = 1 - t * t;
            }
        }
        return res;
    }


    double relu(double x) {
        return x > 0 ? x : 0;
    }
    double reluDerivative(double x) {
        return x > 0 ? 1 : 0;
    }

    Matrix relu(const Matrix &m) {
        Matrix res(m.getRows(), m.getCols());
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m.getRows(); i++) {
            for (int j = 0; j < m.getCols(); j++) {
                res(i,j) = relu(m(i,j));
            }
        }
        return res;
    }

    Matrix reluDerivative(const Matrix &m) {
        Matrix res(m.getRows(), m.getCols());
        for (int i = 0; i < m.getRows(); i++) 
            for (int j = 0; j < m.getCols(); j++)
                res(i, j) = reluDerivative(m(i,j));
        return res;
    }

    Matrix softmax(const Matrix &input) {
        Matrix res(input.getRows(), input.getCols());
        for (size_t i = 0; i < input.getRows(); i++) {
            double rowMax = -std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < input.getCols(); j++) 
                rowMax = std::max(rowMax, input(i, j));
            

            double sumExp = 0.0;
            for (size_t j = 0; j < input.getCols(); j++) {
                res(i, j) = std::exp(input(i, j) - rowMax); 
                sumExp += res(i, j);
            }

            for (size_t j = 0; j < input.getCols(); j++) 
                res(i, j) /= sumExp;
            
        }
        return res;
    }

   Matrix softmaxDerivative(const Matrix &input) {
        Matrix softmax_output = softmax(input);
        Matrix res(input.getRows(), input.getCols());
        
        for (size_t i = 0; i < input.getRows(); i++) {
            for (size_t j = 0; j < input.getCols(); j++) {
                double sj = softmax_output(i,j);
                res(i,j) = sj * (1.0 - sj);
            }
        }
        return res;
    }
}