# Automatic Differentiation in C++

This project provides a C++ library for automatic differentiation, supporting both forward mode and reverse mode.

### Features
- Forward Mode AD: Efficiently computes gradients and Jacobians for functions with a number of outputs larger than the number of inputs.
- Reverse Mode AD: Ideal for computing gradients of functions where the number of inputs is much larger than the number of outputs.
  - Arena allocator: An arena allocator is employed to speed up the computation of gradients and jacobians 
- Eigen Integration: The library specializes certain Eigen classes in order to let the user use Eigen's Vectors and Matrices of `Var` and `DualVar`.
- Cuda Support: The forward mode implementation supports Cuda in order to accelerate the computation of gradients and jacobians.

### Prerequisites
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
    // Gradient computation
    std::function<DualVar<double>(DualVec<double>)> f_grad = ...;
    Eigen::VectorXd<double> point(2);
    point << 2.0, 3.0;
    RealVec<double> grad = gradient(f_grad, point);

    // jacobian computation
    std::function<DualVec<double>(DualVec<double>)> f_jac = ...;

    RealVec<double> f_x(2);  
    JacType<double> jac(2, 2);

    // Jacobian is saved in jac
    jacobian(f, point, f_x, jac); 

    return 0;
}
```

### Reverse Mode
Unlike forward mode, reverse mode builds a computational graph in memory at runtime. In order to let the library manage this computational graph it is advised to follow the examples below.
#### Gradient computation
The `gradient` function accepts a free function/lambda with the following constraints:
- InputType: `Eigen::Vector<Var<double>, Eigen::Dynamic> const &`
- OutputType: `Var<double>`
```c++
#include <iostream>
#include <Eigen/Core>
#include "Var.hpp"
#include "ReverseUtility.hpp"

// Define some useful type aliases
using autodiff::reverse::gradient;
using Var    = autodiff::reverse::Var<double>;
using VecVar = Eigen::Vector<Var, Eigen::Dynamic>;
using Vec    = Eigen::Vector<double, Eigen::Dynamic>;

int main() {
    constexpr size_t N = 10;

    auto f = [N](VecVar const & x) -> Var {
        return x.norm();
    };

    Vec x = 2*Vec::Ones(N);
    double f_x;
    Vec grad;
    
    gradient(f, x, f_x, grad);

    std::cout << " grad(x) = \n" << grad << std::endl;
    std::cout << " f(x) = " << f_x << std::endl;

    return 0;
}
```
Compile with `g++ -o test -I/path/to/eigen3 test.cpp`

#### Jacobian computation
The `jacobian` function accepts a free function/lambda with the following constraints:
- InputType: `Eigen::Vector<Var<double>, Eigen::Dynamic> const &`
- OutputType: `Eigen::Vector<Var<double>, Eigen::Dynamic>`
```c++
#include <iostream>
#include <Eigen/Core>
#include "Var.hpp"
#include "ReverseUtility.hpp"

// Define some useful type aliases
using autodiff::reverse::jacobian;
using Var = autodiff::reverse::Var<double>;
using VecVar = Eigen::Vector<Var, Eigen::Dynamic>;
using Vec = Eigen::Vector<double, Eigen::Dynamic>;
using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

int main() {
    constexpr size_t M = 3;
    constexpr size_t N = 2;

    auto f = [M,N](VecVar const & x) -> VecVar {
        VecVar res(M);

        res <<
            x.norm(),
            x(0) + exp(x(1)),
            x(0) + log(x(1));

        return res;
    };

    Vec x = 2*Vec::Ones(N);
    Vec f_x;
    Mat jac;
    
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
If you don't have all the dependencies already installed, and you want to create a separate virtual env, you can use Docker.
```bash
$ docker build -t autodiff-test .
$ docker run -it autodiff-test
```

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

## Resources
- [The Stan Math Library: Reverse-Mode Automatic Differentiation in C++](https://arxiv.org/abs/1509.07164)
- [autodiff](https://github.com/autodiff/autodiff)
