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

  using NumericType = autodiff::forward::DualVar<T>;
  using NumVecType = Eigen::Matrix<NumericType, Eigen::Dynamic, 1>;
  using RealVec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using JacType = Eigen::MatrixXd;
  using NLSType = std::function<NumVecType(const NumVecType &)>;

};

} // namespace autodiff

#endif