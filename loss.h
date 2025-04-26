#ifndef LOSS_H
#define LOSS_H
#include "matrix.h"
namespace galanet::loss {
        double meanSquaredError(const Matrix &predictions, const Matrix &targets);
        Matrix meanSquaredErrorDerivative(const Matrix &predictions, const Matrix &targets);
        double meanAbsoluteError(const Matrix &predictions, const Matrix &targets);
        Matrix meanAbsoluteErrorDerivative(const Matrix &predictions, const Matrix &targets);
        double crossEntropyLoss(const Matrix &predictions, const Matrix &targets);
        Matrix crossEntropyLossDerivative(const Matrix &predictions, const Matrix &targets);
}
#endif