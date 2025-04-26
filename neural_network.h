#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "matrix.h"
#include "weights_initializer.h"

#include <memory>
#include <string>
namespace galanet {
    class DenseLayer{
        public:
            DenseLayer(int in_dim, int out_dim, std::string activation_name, std::string weight_init_name, double learning_rate);
            Matrix forward(const Matrix &inputs);
            Matrix backward(Matrix &grad);
        protected:
            int in_dim;
            int out_dim;
            double learning_rate;
            Matrix last_inputs;
            Matrix pre_activation;  
            std::string weight_init_name;
            std::string activation_name;
            Matrix weights;
            Matrix bias;
    };
    class NN {
        public: 
            NN(std::string loss_name) ;
            void add_layer(std::unique_ptr<DenseLayer> layer);
            void train(const Matrix &features, const Matrix &targets, const Matrix &val_features = Matrix(0,0), const Matrix &val_targets = Matrix(0,0), int epochs=10, int batchSize = 48, int patience = 5);
            Matrix predict(const Matrix &features);
            double calc_accuracy(const Matrix& features, const Matrix& targets);
            double calculate_loss(const Matrix& predictions, const Matrix& targets);
            Matrix calculate_loss_derivative(const Matrix& predictions, const Matrix& targets);
        private:
            std::vector<std::unique_ptr<DenseLayer>> layers;
            std::string loss_name;
    };
}
#endif