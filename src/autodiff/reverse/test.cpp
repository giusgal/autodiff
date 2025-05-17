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

template <typename T>
T f(T x, T y, T z) {
    return exp(x*y);
}

VarD f(VecVarD x) {
    // return exp(x(0)*x(1));
    // return sqrt(x(0));
    return x.norm();
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
        VecVarD x(3);
        x(0) = VarD(3.0);
        x(1) = VarD(1.0);
        x(2) = VarD(5.0);

        VarD u = f(x);
        u.backward();

        std::cout << "AUTODIFF: \n";
        std::cout << " value: " << u.value() << std::endl;
        std::cout << " dx: " << x(0).grad() << std::endl;
        std::cout << " dy: " << x(1).grad() << std::endl;
        std::cout << " dz: " << x(2).grad() << std::endl;
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
