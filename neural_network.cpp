#include <stdexcept>
#include <iostream>
#include <limits>

#include "neural_network.h"
#include "weights_initializer.h"
#include "activation.h"
#include "loss.h"

namespace galanet{
    DenseLayer::DenseLayer(int in_dim, int out_dim, std::string activation_name, std::string weight_init_name, double learning_rate)
    {
        this->in_dim = in_dim;
        this->out_dim = out_dim;
        this->activation_name = activation_name;
        this->learning_rate=learning_rate;

        if (weight_init_name == "zeros") {
            this->weights = weight_initializers::zeros(in_dim, out_dim);
        } else if (weight_init_name == "random_uniform") {
            this->weights = weight_initializers::random_uniform(in_dim, out_dim);
        } else if (weight_init_name == "he") {
            this->weights = weight_initializers::he_uniform(in_dim, out_dim);
        } else if (weight_init_name == "xavier") {
            this->weights = weight_initializers::xavier_uniform(in_dim, out_dim);
        } else if (weight_init_name == "ones") {
            this->weights = weight_initializers::ones(in_dim, out_dim);
        } else {
            throw std::invalid_argument("Invalid weight initializer");
        }
        this->bias = Matrix(1, out_dim, 0);
    }
    Matrix DenseLayer::forward(const Matrix &inputs)
    {
        last_inputs = inputs;
        pre_activation = inputs * weights;
        for (int i = 0; i < pre_activation.getRows(); i++)
            for (int j = 0; j < pre_activation.getCols(); j++)
                pre_activation(i, j) += this->bias(0, j);
        if (this->activation_name == "relu")
            return galanet::activation::relu(pre_activation);
        else if (this->activation_name == "tanh")
            return galanet::activation::tanh(pre_activation);
        else if (this->activation_name == "softmax")
            return galanet::activation::softmax(pre_activation);
        else throw std::invalid_argument("Invalid activation function");    
    }
    Matrix DenseLayer::backward(Matrix &grad){
        Matrix gradActivation;
        if (this->activation_name == "relu")
            gradActivation =  galanet::activation::reluDerivative(pre_activation);
        else if (this->activation_name == "tanh")
            gradActivation =  galanet::activation::tanhDerivative(pre_activation);
        else if (this->activation_name == "softmax")
            gradActivation= galanet::activation::softmaxDerivative(pre_activation);
        else throw std::invalid_argument("Invalid activation function");


        for(int i=0;i<grad.getRows();i++)
            for(int j=0;j<grad.getCols();j++)
                grad(i,j)*=gradActivation(i,j);

        Matrix input_grad = grad * weights.transpose();  //calculate input gradient before weight update


        Matrix weights_grad=last_inputs.transpose()*grad;
        Matrix bias_grad=Matrix(1,grad.getCols(),0);

        for(int i=0;i<grad.getRows();i++)
            for(int j=0;j<grad.getCols();j++)
                bias_grad(0,j)+=grad(i,j);
        weights= weights-weights_grad*learning_rate;
        bias= bias - bias_grad*learning_rate;
       
        return input_grad;
    }




    NN::NN(std::string loss_name)
    {
        this->loss_name = loss_name;
        this->layers=std::vector<std::unique_ptr<DenseLayer>>();
    }
    void NN::add_layer(std::unique_ptr<DenseLayer> layer)
    {
        this->layers.push_back(std::move(layer));
    }

    Matrix NN::predict(const Matrix &features){
        Matrix res=features;
        for(int i=0;i<this->layers.size();i++){
            res=this->layers[i]->forward(res);
        }
        return res;
    }

    void NN::train(const Matrix &features, const Matrix &targets, const Matrix &val_features , const Matrix &val_targets , int epochs, int batchSize, int patience ){
        double best_val_loss = std::numeric_limits<double>::infinity();
        int no_improve = 0;
        for(int i=1;i<=epochs;i++){
            double epoch_loss = 0;
            for(int j=0;j<features.getRows();j+=batchSize){
                Matrix batch_features=features.subset_rows(j,std::min(j+batchSize,features.getRows()));
                Matrix batch_targets=targets.subset_rows(j,std::min(j+batchSize,features.getRows()));

                Matrix pred=predict(batch_features);
                Matrix grad=calculate_loss_derivative(pred,batch_targets);

                for(int k=this->layers.size()-1;k>=0;k--){
                    grad=this->layers[k]->backward(grad);
                }
                double batch_loss=calculate_loss(pred, batch_targets);
                epoch_loss += batch_loss;
                int progress = (j * 100) / features.getRows();
                if (progress % 10 == 0 && (j == 0 || (j * 100) / features.getRows() != ((j - batchSize) * 100) / features.getRows())) {
                    std::cout << "Epoch Progress: " << progress << "% - Batch " << std::min(j + batchSize, features.getRows()) 
                              << "/" << features.getRows() << " - Loss: " << batch_loss << "\n";
                }
            }
            epoch_loss /= (features.getRows() / batchSize);
            Matrix val_predictions=predict(val_features);
            double val_loss = calculate_loss(val_predictions, val_targets);
            
            // Early stopping
            if(val_loss < best_val_loss) {
                best_val_loss = val_loss;
                no_improve = 0;
            } else {
                no_improve++;
                if(no_improve >= patience) break;
            }

            std::cout << "Epoch " << i << "/" << epochs 
                  << " - loss: " << epoch_loss / (features.getRows()/batchSize)
                  << " - val_loss: " << val_loss 
                  << " - val_accuracy: " << calc_accuracy(val_predictions, val_targets) << "\n";

        }
    }

    Matrix NN::calculate_loss_derivative(const Matrix& predictions, const Matrix& targets){
        if(this->loss_name=="cross_entropy")
            return galanet::loss::crossEntropyLossDerivative(predictions,targets);
        else if(this->loss_name=="mean_squared_error" || this->loss_name=="mse")
            return galanet::loss::meanSquaredErrorDerivative(predictions,targets);
        else if(this->loss_name=="mean_absolute_error" || this->loss_name=="mae")
            return galanet::loss::meanAbsoluteErrorDerivative(predictions,targets);
        else throw std::invalid_argument("Invalid loss function");
    }

    double NN::calculate_loss(const Matrix& predictions, const Matrix& targets){
        if(this->loss_name=="cross_entropy")
            return galanet::loss::crossEntropyLoss(predictions,targets);
        else if(this->loss_name=="mean_squared_error" || this->loss_name=="mse")
            return galanet::loss::meanSquaredError(predictions,targets);
        else if(this->loss_name=="mean_absolute_error" || this->loss_name=="mae")
            return galanet::loss::meanAbsoluteError(predictions,targets);
        else throw std::invalid_argument("Invalid loss function");
    }
    

    double NN::calc_accuracy(const Matrix& pred, const Matrix& targets){
        int t = 0;
        #pragma omp parallel for reduction(+:t)
        for(int i = 0; i < pred.getRows(); i++){
            int pred_index = 0;
            int target_index = 0;
            for(int j = 0; j < pred.getCols(); j++){
                if(pred(i, j) > pred(i, pred_index))
                    pred_index = j;
                if(targets(i, j) == 1)
                    target_index = j;
            }
            if(pred_index == target_index)
                t++;
        }

        return (double)t / pred.getRows();
    }
}