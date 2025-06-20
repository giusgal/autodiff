#ifndef JACOBIAN_TRAITS_HPP_
#define JACOBIAN_TRAITS_HPP_

#include <Eigen/Dense>
#include <vector>
#include <functional>
#include "DualVar.hpp"

namespace autodiff {
namespace forward {

template<typename T>
struct JacobianTraits {
public:

  using dv = autodiff::forward::DualVar<T>;

  using FwArgType = Eigen::Matrix<autodiff::forward::DualVar<T>, Eigen::Dynamic, 1>;
  using FwRetType = FwArgType;
  using FwNLSType = std::function<FwRetType(const FwArgType &)>;

  using CudaArgType = FwArgType;
  using CudaRetType = autodiff::forward::DualVar<T>;
  using CudaDeviceFn = autodiff::forward::DualVar<T> (*)(const FwArgType&, const int out_idx);

  using RealVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using JacType = Eigen::MatrixXd;

  using NLSType = FwNLSType;

};

}; // namespace autodiff
}; // namespace forward

#endif // JACOBIAN_TRAITS_HPP_