# Model Implementations and Tests

## Overview

This part illustrates how we can use forward-model automatic differentiation to build an example product, typically a model that can approximate an arbitrary function.

## ✨ Classes 

* **Linear Model:** Implementations of forward and reverse mode autodiff primitives.
* **Parallel Linear Model:** Simple implementations of Linear and basic Feedforward Neural Network models.
* **Optimizers:** Gradient-based optimization algorithms including Stochastic Gradient Descent (SGD) and Adam.
* **Neural Model:** A basic implementation of Newton's method for root finding/optimization (likely uses autodiff for Jacobian/Hessian).
* **Cache-Optimized Neural Model:** Written using modern C++ standards (likely C++20 based on CMake).
* **Parallel Neural Model:** Leverages the Eigen library for efficient vector and matrix operations (indicated by `EigenSupport.hpp`).
![One Layer Deep Learning](model_comparison_hidden_sweep.png)
 

## 🏗️ Implementations and Tests

### Linear Model and Parallel Linear Model

1.  Clone the repository:
    ```bash
    git clone <repository_url> # Replace with your repo URL
    cd <project_directory>
    ```
2.  Create a build directory and navigate into it:
    ```bash
    mkdir build
    cd build
    ```
3.  Run CMake to configure the project and generate build files:
    ```bash
    cmake ..
    # For a Debug build (common when developing):
    # cmake -DCMAKE_BUILD_TYPE=Debug ..
    ```
    *(Note: If Eigen is not found by CMake, you might need to provide its path using `-DEigen3_DIR=/path/to/eigen3/cmake` or similar, depending on how Eigen was installed).*
4.  Build the project using your chosen build tool (typically `make` or `ninja`):
    ```bash
    make
    # Or:
    # ninja
    ```

This will compile the library, examples, and tests.

## 🏃 Running Examples

After building the project, the executables for the examples will be located in your build directory (e.g., `./build/`).

Navigate to the build directory and run the examples:

```bash
cd build
./linear_neural_comparison
./NeuralHiddenSizeComparison
./newton_test
# ... and any other example executables you have
