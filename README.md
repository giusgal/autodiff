# C++ Automatic Differentiation Library

## Prerequisites
* A C++20 compliant compiler
* CMake (>=3.28)
* The Eigen library (>=3.4.0)
* Cuda (optional)

## Examples
### Forward Mode
The following is an example that shows how the forward mode module can be used:
```c++
#include <Eigen/Core>
#include "DualVar.hpp"
#include "ForwardUtility.hpp"

using namespace autodiff::forward;
using DualVar = DualVar<double>;

int main() {
    // gradient computation

    // jacobian computation

    return 0;
}
```

### Reverse Mode
To use the reverse mode module, it is advised to proceed as follows in order to avoid possible runtime problems:
#### Gradient computation
```c++
#include <iostream>
#include <Eigen/Core>
#include "Var.hpp"                // Var
#include "ReverseUtility.hpp"     // gradient, jacobian

// Define some useful type aliases
using Var = autodiff::reverse::Var<double>;
using VecVar = Eigen::Vector<Var, Eigen::Dynamic>;
using Vec = Eigen::Vector<double, Eigen::Dynamic>;
using autodiff::reverse::gradient;

int main() {
    constexpr size_t N = 10;

    // Define your function as a free function/lambda that:
    //  - Returns a Var
    //  - Accepts a reference to a constant VecVar
    auto f = [](VecVar const & x) -> Var {
        return x.norm();
    };

    Vec x = 2*Vec::Ones(N); // input vector
    double f_x;             // function evaluated in x
    Vec grad;               // gradient
    
    gradient(f, x, f_x, grad);

    std::cout << " grad(x) = \n" << grad << std::endl;
    std::cout << " f(x) = " << f_x << std::endl;

    return 0;
}
```
Compile with `g++ -o test -I/path/to/eigen3 test.cpp`

#### Jacobian computation
```c++
#include <iostream>
#include <Eigen/Core>
#include "Var.hpp"                // Var
#include "ReverseUtility.hpp"     // gradient, jacobian

// Define some useful type aliases
using Var = autodiff::reverse::Var<double>;
using VecVar = Eigen::Vector<Var, Eigen::Dynamic>;
using Vec = Eigen::Vector<double, Eigen::Dynamic>;
using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
using autodiff::reverse::jacobian;

int main() {
    constexpr size_t M = 3;
    constexpr size_t N = 2;

    // Define your function as a free function/lambda that:
    //  - Returns a VecVar
    //  - Accepts a reference to a constant VecVar
    auto f = [M,N](VecVar const & x) -> VecVar {
        VecVar res(M);

        res <<
            x.norm(),
            x(0) + exp(x(1)),
            x(0) + log(x(1));

        return res;
    };

    Vec x = 2*Vec::Ones(N);   // input vector
    Vec f_x;                  // function evaluated in x
    Mat jac;                  // jacobian
    
    jacobian(f, x, f_x, jac);

    std::cout << " jac(x) = \n" << jac << std::endl;
    std::cout << " f(x) = \n" << f_x << std::endl;

    return 0;
}
```
Compile with `g++ -o test -I/path/to/eigen3 test.cpp`

## Building
### Local build
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
    To enable CUDA computation, pass the following option to CMake:
    ```bash
    cmake -DENABLE_CUDA=ON ..
    ```
4.  Build the project
    ```bash
    make
    ```
This will compile the library, examples, and tests.

### Build using Docker
If you don't have all the dependencies already installed, and you want to create a separate virtual env, you can check the following:
If you are using Linux, run_interactive.sh will help to build a Docker environment. 
1. Make sure you are in the root folder of this project, autodiff
```
chmod u+x run_interactive.sh
./run_interactive
```
2. Then you can take a break. After some minutes, you will find yourself inside the Docker container!
   You can either run the tests or create the build folder in the project folder and start developing.

## Running Examples
Navigate to the build directory and run the examples:

```bash
cd build
./linear_neural_comparison
./NeuralHiddenSizeComparison
./newton_test
# ... and any other example executables you have
```

## Running Tests
Navigate to the build directory and run the tests:

```bash
cd build
ctest
```

## Plotting results
Some tests generate CSV files; for some, there are corresponding Python scripts to plot the CSV results. You can call those scripts like the following (using Python virtual environments):
```
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib pandas
python plot_models.py
deactivate
```
