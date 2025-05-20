#include <iostream>

#include <Eigen/Core>
#include "Var.hpp"
#include "DerivativeUtility.hpp"

using Var = autodiff::reverse::Var<double>;
using VecVar = Eigen::Vector<Var, Eigen::Dynamic>;
using Vec = Eigen::Vector<double, Eigen::Dynamic>;
using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

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

Var f_N1(VecVar x) {
    // return exp(x(0)*x(1));
    // return sqrt(x(0));
    // return x.norm();
    return x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
}

VecVar f_NM(VecVar x) {
    Var y1 = x.norm();
    Var y2 = exp(x(0)*x(1));
    // VarD y3 = x(0)*x(0) + x(1)*x(1) + x(2)*x(2);

    VecVar y(2);
    y << y1, y2;

    return y;
}

/*
 * TODO:
 *  1. Refactor Var/NodeManager to simplify the creation of new Node(s)
 *     (that 'template' keyword in every overloaded operator in Var
 *     is ugly) and remove the need for a manager_ptr_ member variable in Var
 *  Then:
 *  2. Implement more operators
 *  3. Implement a memory pool (arena allocator)
 * */
int main() {
    {
        Vec x(3);
        x << 3.0, 1.0, 5.0;
        Vec f_x;
        Mat jac;

        jacobian(f_NM, x, f_x, jac);

        std::cout << jac << std::endl;
    }

    // {
    //     // Gradient-descent
    //     constexpr size_t N = 10;
    //     double f_x = 0.0;
    //     Vec grad(3);
    //     double lambda = 0.8;
    //     Vec x(3);
    //     x << 1.0, 1.0, 1.0;
    //
    //     for(size_t i = 0; i < N; ++i) {
    //         gradient(f_N1, x, f_x, grad);
    //         x = x - lambda*grad;
    //         std::cout << "Iteration #" << i << "\n"
    //             << " x: ";
    //         std::cout << x << "\n\n";
    //     }
    //
    //     std::cout << "AUTODIFF: \n";
    //     std::cout << " value: " << f_x << std::endl;
    //     std::cout << " dx: " << grad(0) << std::endl;
    //     std::cout << " dy: " << grad(1) << std::endl;
    //     std::cout << " dz: " << grad(2) << std::endl;
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
