#include <chrono>
#include <iostream>

#include <Eigen/Core>
#include "Var.hpp"
#include "DerivativeUtility.hpp"

using Var = autodiff::reverse::Var<double>;
using VecVar = Eigen::Vector<Var, Eigen::Dynamic>;
using Vec = Eigen::Vector<double, Eigen::Dynamic>;
using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

constexpr size_t N = 100;

// std::array<double, 3>
// finite_diff(
//     std::function<double(double, double, double)> f,
//     std::array<double, 3> const & input,
//     double h
// ) {
//     std::array<double, 3> output{};
//
//     output[0] =
//         (f(input[0]+h,input[1],input[2])-f(input[0]-h,input[1],input[2]))/(2*h);
//     output[1] =
//         (f(input[0],input[1]+h,input[2])-f(input[0],input[1]-h,input[2]))/(2*h);
//     output[2] =
//         (f(input[0],input[1],input[2]+h)-f(input[0],input[1],input[2]-h))/(2*h);
//
//     return output;
// }

template <typename T>
T f_31(T x, T y, T z) {
    return exp(x*y);
}

Var f_N1(VecVar const & x) {
    // return exp(x(0)*x(1));
    // return sqrt(x(0));
    return sqrt(x.sum());
    // return x.norm();
    // return x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
}

VecVar f_NM(VecVar x) {
    VecVar y(N);

    for(size_t i = 0; i < N; ++i) {
        y(i) = x.norm();
    }

    // Var y1 = x.norm();
    // Var y2 = exp(x(0)*x(1));
    // VarD y3 = x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
    // y << y1, y2;

    return y;
}

int main() {
    {
        Vec x = Vec::Ones(N);
        Vec f_x;
        Mat jac;
    
        auto t1 = std::chrono::high_resolution_clock::now();
        jacobian(f_NM, x, f_x, jac);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "test function took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds\n";
        // std::cout << jac << std::endl;
    }

    // {
    //     Vec x = Vec::Ones(N);
    //     double f_x;
    //     Vec grad;

    //     auto t1 = std::chrono::high_resolution_clock::now();
    //     gradient(f_N1, x, f_x, grad);
    //     auto t2 = std::chrono::high_resolution_clock::now();
    //     std::cout << "test function took "
    //           << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
    //           << " milliseconds\n";
    //     // std::cout << jac << std::endl;
    // }

    // {
    //     // Not recommended usage
    //     //  Why?
    //     //   gradient() and jacobian() not only compute
    //     //   the gradient and the jacobian respectively, but
    //     //   they also manage memory to avoid memory leaks.
    //     Var x(3.0);
    //     Var y(1.0);
    //     Var z(5.0);
    //
    //     Var u = f_31(x,y,z);
    //     u.backward();
    //
    //     std::cout << "AUTODIFF: \n";
    //     std::cout << " value: " << u.value() << std::endl;
    //     std::cout << " dx: " << x.grad() << std::endl;
    //     std::cout << " dy: " << y.grad() << std::endl;
    //     std::cout << " dz: " << z.grad() << std::endl;
    // }
    
    return 0;
}
