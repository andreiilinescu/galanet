# GalaNet - Custom Neural Network in C++ implementation

GalaNet is my my implementation of a neural network in c++. It is built upon a custom Linear Algebra Operations library (available in `matrix.cpp`), created by me from scratch. 

Thea idea of this project came after I finished the Compuatitonal Intelligence and the Concepts of Programming Languages courses. I wanted to take a stab at manually implementing a neural network, using parallel operations from the `OpenMP` library.


## Key Features
- **Dense Layers:** Customizable with multiple activation functions (ReLU, Tanh, Softmax).
- **Flexible Loss Functions:** Mean Squared Error (MSE), Mean Absolute Error (MAE), Cross-Entropy.
- **Robust Initialization:** Implements He, Xavier/Glorot, and Random Uniform initializations.
- **Parallelization:** Optimized matrix operations leveraging OpenMP.
- **Custom Linear Algebra Library:** Fully self-built matrix operations in `matrix.cpp`, featuring all essential linear algebra functionalities.
- **Training Enhancements:** Includes batch training and early stopping to prevent overfitting.
- **Dataset Support:** Integrated MNIST dataset loader for easy experimentation.

## Architecture & Usage
GalaNet maintains a modular architecture, making it straightforward to experiment with and expand. Whether you're exploring neural networks academically or practically, GalaNet provides an intuitive playground to deepen your understanding.

Check out `example_mnist.cpp` for a practical example of how GalaNet can be used.

## Dependencies
GalaNet was intentionally developed with minimal dependencies, relying solely on:
- **C++17** (Standard Template Library - STL)
- **OpenMP** (for parallel processing)

No external libraries were used, making GalaNet lightweight and ideal for educational purposes and experiments.

## Build Instructions
Simply clone the repository and build using your favorite C++17 compatible compiler with OpenMP support enabled:

```bash
git clone https://github.com/andreiilinescu/galanet.git
cd galanet
make
```
