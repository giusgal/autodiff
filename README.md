# C++ Automatic Differentiation Library

## Prerequisites

* A C++20 compliant compiler
* CMake (>=3.28)
* The Eigen library (>=3.4.0)
* Cuda (optional)

## Building

1.  Clone the repository
2.  Create a build directory and navigate into it:
    ```bash
    mkdir build
    cd build
    ```
3.  Run CMake to configure the project and generate build files:
    ```bash
    cmake ..
    ```
4.  Build the project
    ```bash
    make
    ```

This will compile the library, examples, and tests.

## Running Examples

Navigate to the build directory and run the examples:

```bash
cd build
./linear_neural_comparison
./NeuralHiddenSizeComparison
./newton_test
# ... and any other example executables you have
```