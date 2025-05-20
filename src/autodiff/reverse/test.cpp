#include <iostream>
#include <functional>
#include <array>

#include <Eigen/Core>
#include "EigenSupport.hpp"
#include "Tape.hpp"

using namespace autodiff::reverse;
using VarD = Var<double>;
using TapeD = Tape<double>;
using VecVarD = Eigen::Vector<VarD, Eigen::Dynamic>;
using Vec = Eigen::Vector<double, Eigen::Dynamic>;

std::array<double, 3>
finite_diff(
    std::function<double(double, double, double)> f,
    std::array<double, 3> const & input,
    double h
) {
    std::array<double, 3> output{};

    output[0] =
        (f(input[0]+h,input[1],input[2])-f(input[0]-h,input[1],input[2]))/(2*h);
    output[1] =
        (f(input[0],input[1]+h,input[2])-f(input[0],input[1]-h,input[2]))/(2*h);
    output[2] =
        (f(input[0],input[1],input[2]+h)-f(input[0],input[1],input[2]-h))/(2*h);

    return output;
}

// template <typename T>
// T f(T x, T y, T z) {
//     return exp(x*y);
// }

VarD f(VecVarD x) {
    // return exp(x(0)*x(1));
    // return sqrt(x(0));
    // return x.norm();
    return x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
}

void gradient(std::function<VarD(VecVarD)> f, Vec const & x, double & f_x, Vec & grad) {
    VecVarD var_x(x.size());
    
    for(size_t i = 0; i < var_x.size(); ++i) {
        var_x(i) = VarD(x(i));
    }

    VarD y = f(var_x);
    y.backward();

    f_x = y.value();

    grad.resizeLike(var_x);
    for(size_t i = 0; i < grad.size(); ++i) {
        grad(i) = var_x(i).grad();
    }

    TapeD::instance().manager().clear();
}

int main() {

    // double out = f(x.value(), y.value(), z.value());
    // auto [dx, dy, dz] = finite_diff(
    //     f<double>,
    //     {x.value(), y.value(), z.value()},
    //     0.00001
    // );
    //
    // std::cout << "ORIGINAL: \n";
    // std::cout << " value: " << out << std::endl;
    // std::cout << " dx: " << dx << std::endl;
    // std::cout << " dy: " << dy << std::endl;
    // std::cout << " dz: " << dz << std::endl;

    {

        constexpr size_t N = 10;
        Vec x(3);
        x << 1.0, 1.0, 1.0;
        double f_x = 0.0;
        Vec grad(3);
        double lambda = 0.8;

        for(size_t i = 0; i < N; ++i) {
            gradient(f, x, f_x, grad);
            x = x - lambda*grad;
            std::cout << "Iteration #" << i << "\n"
                << " x: ";
            std::cout << x << "\n\n";
        }


        std::cout << "AUTODIFF: \n";
        std::cout << " value: " << f_x << std::endl;
        std::cout << " dx: " << grad(0) << std::endl;
        std::cout << " dy: " << grad(1) << std::endl;
        std::cout << " dz: " << grad(2) << std::endl;
    }

    // {
    //     VarD x(3.0);
    //     VarD y(1.0);
    //     VarD z(5.0);
    //
    //     VarD u = f(x,y,z);
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
