#pragma once

#include <Eigen/Dense>
#include <vector>
#include <functional>
#include "DualVar.hpp"
#include "Var.hpp"

namespace newton {

struct JacobianTraits {
public:
    // Forward traits
    using dv = autodiff::forward::DualVar<double>;

    using FwArgType = Eigen::Matrix<dv, Eigen::Dynamic, 1>;
    using FwRetType = FwArgType;
    using FwNLSType = std::function<FwRetType(const FwArgType &)>;

    using CudaArgType = FwArgType;
    using CudaRetType = dv;
    using CudaDeviceFn = dv (*)(const FwArgType&, const int out_idx);

    // Reverse traits
    using var = autodiff::reverse::Var<double>;

    using RvArgType = Eigen::Matrix<var, Eigen::Dynamic, 1>;
    using RvRetType = RvArgType;
    using RvNLSType = std::function<RvRetType(RvArgType const &)>;

    // "double" traits
    using RealVec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using JacType = Eigen::MatrixXd;

};

}; // namespace newton
