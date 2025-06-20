// Differentiator.h
#ifndef __DIFFERENTIATOR__CPP__
#define __DIFFERENTIATOR__CPP__

#include <vector>
#include <functional>
#include <Eigen/Dense>

#include "DualVar.hpp"


namespace autodiff {
namespace forward {

using dv = DualVar<double>;
using dvec = std::vector<dv>;;
using RealVec = std::vector<double>;

template <typename T>
DualVar<T> derivative(std::function<dv(dv)>f, double x0){
    DualVar<T> res = f(DualVar<T>(x0, 1.0));
    return res;
}


std::vector<double> gradient(std::function<dv(dvec)>f, std::vector<double> x) {

    dvec xd;
    std::vector<double> res;

    xd.reserve(size(x));
    res.reserve(size(x));

    for(int i = 0; i < size(x); i++){
        xd.push_back(DualVar(x[i], 0.0));
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
