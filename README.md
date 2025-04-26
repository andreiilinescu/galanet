# GalaNet - A Personal Journey into Neural Networks with C++

GalaNet isn't just another neural network library; it's a reflection of curiosity and an exercise in applying theoretical knowledge practically. Born out of nostalgia for C++ and inspired by the concepts I explored during my Computational Intelligence and Machine Learning courses, GalaNet is a lightweight neural network library dedicated to efficiently solving the MNIST digit classification problem.

After exploring OpenMP in my Concepts of Programming Languages course, I became intrigued by parallel computing. GalaNet allowed me to experiment firsthand with OpenMP, enhancing performance through parallelized matrix operations. Additionally, I challenged myself by building a custom linear algebra operations library from scratch in `matrix.cpp`, completely self-contained and optimized for GalaNet.

GalaNet is my way of revisiting C++ and bridging theory with practice—perhaps it'll be your stepping stone to something exciting too.

## 🚀 Key Features
- **Dense Layers:** Customizable with multiple activation functions (ReLU, Tanh, Softmax).
- **Flexible Loss Functions:** Mean Squared Error (MSE), Mean Absolute Error (MAE), Cross-Entropy.
- **Robust Initialization:** Implements He, Xavier/Glorot, and Random Uniform initializations.
- **Parallelization:** Optimized matrix operations leveraging OpenMP.
- **Custom Linear Algebra Library:** Fully self-built matrix operations in `matrix.cpp`, featuring all essential linear algebra functionalities.
- **Training Enhancements:** Includes batch training and early stopping to prevent overfitting.
- **Dataset Support:** Integrated MNIST dataset loader for easy experimentation.

## 📚 Architecture & Usage
GalaNet maintains a modular architecture, making it straightforward to experiment with and expand. Whether you're exploring neural networks academically or practically, GalaNet provides an intuitive playground to deepen your understanding.

Check out `example_mnist.cpp` for a practical example of how GalaNet can be used.

## 🛠️ Dependencies
GalaNet was intentionally developed with minimal dependencies, relying solely on:
- **C++17** (Standard Template Library - STL)
- **OpenMP** (for parallel processing)

No external libraries were used, making GalaNet lightweight and ideal for educational purposes and experiments.

## 🏗️ Build Instructions
Simply clone the repository and build using your favorite C++17 compatible compiler with OpenMP support enabled:

```bash
git clone https://github.com/andreiilinescu/galanet.git
cd galanet
make
```
