#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <algorithm>
#include <numeric>
#include <random>
#include <ctime>


#include "matrix.h"
#include "neural_network.h" 
using namespace galanet;

galanet::Matrix mnistImagesToMatrix(const std::string &path) {
    std::ifstream file (path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        exit(1);
    }
    int magic = 0;
    int num_items = 0;
    int num_rows = 0;
    int num_cols = 0;
    file.read((char*)&magic, sizeof(magic));
    magic = __builtin_bswap32(magic);
    if(magic!=2051)
        throw std::invalid_argument("Invalid MNIST file (magic number)"); 
    
    file.read((char*)&num_items, sizeof(num_items));
    num_items = __builtin_bswap32(num_items);
    file.read((char*)&num_rows, sizeof(num_rows));
    num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, sizeof(num_cols));
    num_cols = __builtin_bswap32(num_cols);
    galanet::Matrix res(num_items, num_rows * num_cols);
    for (int i = 0; i < num_items; i++) {
        for (int j = 0; j < num_rows * num_cols; j++) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, sizeof(pixel));
            res(i, j) = pixel;
        }
    }
    return res;
}

galanet::Matrix mnistLabelsToMatrix(const std::string &path) {
    std::ifstream file (path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        exit(1);
    }
    int magic = 0;
    int num_items = 0;
    file.read((char*)&magic, sizeof(magic));
    magic = __builtin_bswap32(magic);
    if(magic!=2049)
        throw std::invalid_argument("Invalid MNIST file (magic number)"); 
    file.read((char*)&num_items, sizeof(num_items));
    num_items = __builtin_bswap32(num_items);
    galanet::Matrix res(num_items, 10);
    for (int i = 0; i < num_items; i++) {
        unsigned char label = 0;
        file.read((char*)&label, sizeof(label));
        res(i, label) = 1;
    }
    return res;
}
int main(){
    try {
    srand(84);//set seed
    //load training Data
    Matrix training_set = mnistImagesToMatrix("./mnist_data/train-images.idx3-ubyte");
    std::cout << "Training set shape: " << training_set.getRows() << "x" << training_set.getCols() << "\n";
    training_set=training_set/255.0;
    Matrix labels = mnistLabelsToMatrix("./mnist_data/train-labels.idx1-ubyte"); 
    std::cout << "Labels shape: " << labels.getRows() << "x" << labels.getCols() << "\n";
    //load test Data
    Matrix test_set = mnistImagesToMatrix("./mnist_data/t10k-images.idx3-ubyte"); 
    Matrix test_labels= mnistLabelsToMatrix("./mnist_data/t10k-labels.idx1-ubyte"); 
    double learning_rate=0.0001;
    NN  nn=NN("cross_entropy"); //create neural network;

    DenseLayer layer1(784, 128, "relu", "he", learning_rate); //create first layer
    DenseLayer layer2(128, 10, "softmax", "random_uniform", learning_rate); //create second layer
    nn.add_layer(std::make_unique<DenseLayer>(layer1));
    nn.add_layer(std::make_unique<DenseLayer>(layer2));
    std::cout<<"created\n";
    nn.train(training_set, labels, training_set, labels, 20, 64); // Train neural network

    Matrix test_pred=nn.predict(test_set);
    std::cout << "Test Accuracy: " << nn.calc_accuracy(test_pred,test_labels) << std::endl;
     } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}