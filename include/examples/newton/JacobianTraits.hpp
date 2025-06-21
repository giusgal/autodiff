#ifndef JACOBIAN_TRAITS_HPP_
#define JACOBIAN_TRAITS_HPP_

#include <Eigen/Dense>
#include <vector>
#include <functional>
#include "DualVar.hpp"

namespace newton {

struct JacobianTraits {
public:

  using dv = autodiff::forward::DualVar<double>;

  using FwArgType = Eigen::Matrix<autodiff::forward::DualVar<double>, Eigen::Dynamic, 1>;
  using FwRetType = FwArgType;
  using FwNLSType = std::function<FwRetType(const FwArgType &)>;

  using CudaArgType = FwArgType;
  using CudaRetType = autodiff::forward::DualVar<double>;
  using CudaDeviceFn = autodiff::forward::DualVar<double> (*)(const FwArgType&, const int out_idx);

  using RealVec = Eigen::Matrix<double, Eigen::Dynamic, 1>;
  using JacType = Eigen::MatrixXd;

};

}; // namespace newton

#endif // JACOBIAN_TRAITS_HPP_