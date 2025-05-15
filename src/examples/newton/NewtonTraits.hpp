#ifndef NEWTON_TRAITS_HPP_
#define NEWTON_TRAITS_HPP_

#include <Eigen/Dense>
#include <vector>
#include <functional>

namespace newton {

// either have dim as tempalte parameter of as vector size
template<typename T>
struct NewtonTraits {
public:

  using dv = autodiff::forward::DualVar<T>;
  using FwArgType = Eigen::Matrix<dv, Eigen::Dynamic, 1>;
  using FwRetType = FwArgType;
  using FwNLSType = std::function<FwArgType(const FwArgType &)>;

  using var = autodiff::reverse::Var<T>;
  using RevArgType = Eigen::Matrix<var, Eigen::Dynamic, 1>;
  using RevRetType = RevArgType;
  using RevNLSType = std::function<FwArgType(const FwArgType &)>;

  using RealVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using JacType = Eigen::MatrixXd;
  using NLSType = FwNLSType;

};

} // namespace autodiff

#endif