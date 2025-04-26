#include <cmath>
#include <iostream>
#include "loss.h"

namespace galanet::loss {
    // Mean Squared Error (MSE)
    double meanSquaredError(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) 
            throw std::invalid_argument("targets and predictions must have the same shape");
        
        return (predictions - targets).pow(2).sum() / (2.0 * predictions.getRows());
    }

       Matrix meanSquaredErrorDerivative(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("targets and predictions must have the same shape");
        }
        return (predictions - targets) / predictions.getRows();  // Removed the 2* due to 1/2n in loss
    }

    // Mean Absolute Error (MAE)
    double meanAbsoluteError(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) 
            throw std::invalid_argument("targets and predictions must have the same shape");
        
        return (predictions - targets).abs().sum() / predictions.getRows();
    }

    Matrix meanAbsoluteErrorDerivative(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("targets and predictions must have the same shape");
        }
        Matrix diff = predictions - targets;
        return diff.sign() / predictions.getRows();
    }

    // Cross-Entropy Loss
     double crossEntropyLoss(const Matrix &predictions, const Matrix &targets) {
        if (predictions.getRows() != targets.getRows() || 
            predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("Shape mismatch");
        }

        double loss = 0.0;
        #pragma omp parallel for reduction(+:loss)
        for (int i = 0; i < predictions.getRows(); i++) {
            for (int j = 0; j < predictions.getCols(); j++) {
                double pred = std::max(std::min(predictions(i,j), 
                                              1.0 - Matrix::EPSILON), 
                                              Matrix::EPSILON);
                loss -= targets(i,j) * std::log(pred);
            }
        }
        return loss / predictions.getRows();
    }

    Matrix crossEntropyLossDerivative(const Matrix &predictions, const Matrix &targets) {
        //assume predicitons are outputs of softmax
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols()) {
            throw std::invalid_argument("targets and predictions must have the same shape");
        }
        return (predictions - targets) / predictions.getRows();
    }
}