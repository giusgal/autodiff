#include <iostream>
#include <functional>
#include <Eigen/Core>
#include "autodiff.hpp"
#include "neural.hpp"


using dv = autodiff::DualVar<double>;
using dvec = std::vector<dv>;



DualVar<double> fun(dv x){
    return 3.0 * log(x) + 2.0;
}

DualVar<double> fun_v(dvec x){
    return 16.0 * x[0] + 32.0 * x[1] + x[2] * x[3];
}


dvec fun_j(dvec x){
    dvec res;
    res.reserve(2);

    res.push_back(5.0 * x[0] * x[0] + x[1] * x[1] * x[0] + sin(2.0 * x[1]) * sin(2.0 * x[1]) - 2.0);
    res.push_back(autodiff::pow(std::exp(1), 2.0 * x[0]-x[1]) + 4.0 * x[1] - 3.0);

    return res;
}


using namespace std;
int main() {

    Eigen::VectorXd v(2);
    v << 1.0, 1.0;

    Eigen::VectorXd res = newton(fun_j, v, 2);
}




