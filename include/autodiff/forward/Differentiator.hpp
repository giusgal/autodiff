// Differentiator.h
#ifndef __DIFFERENTIATOR__CPP__
#define __DIFFERENTIATOR__CPP__

#include <vector>
#include <functional>
#include <Eigen/Dense>

#include "../../../include/autodiff/forward/DualVar.hpp"


namespace autodiff {
namespace forward {

template <typename T>
using dv = DualVar<T>;
template <typename T>
using dvec = Eigen::Matrix<dv, Eigen::Dynamic, 1>;
template <typename T>
using RealVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
DualVar<T> derivative(std::function<dv<T>(dv<T>)>f, double x0){
    DualVar<T> res = f(DualVar<T>(x0, 1.0));
    return res;
}

template <typename T>
std::vector<T> gradient(std::function<dv<T>(dvec<T>)>f, std::vector<T> x) {

    dvec xd;
    std::vector<T> res;

    xd.reserve(size(x));
    res.reserve(size(x));

    for(int i = 0; i < size(x); i++){
        xd.push_back(DualVar<T>(x[i], 0.0));
    }

    for(int i = 0; i < size(x); i++){
        xd[i].setInf(1.0);
        res.push_back(f(xd).getInf());
        xd[i].setInf(0.0);
    }

    return res;
}

}; // namespace forward
}; // namespace autodiff

#endif // __DIFFERENTIATOR__CPP__
