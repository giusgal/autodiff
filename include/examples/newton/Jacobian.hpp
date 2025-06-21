#ifndef JACOBIAN_HPP_
#define JACOBIAN_HPP_

#include <functional>
#include <Eigen/Dense>
#include <omp.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "JacobianTraits.hpp"
#include "ForwardEigenSupport.hpp"
#include "CudaSupport.hpp"
#include "ForwardDifferentiator.hpp"

namespace autodiff {
namespace forward {

template <typename T>
class JacobianBase : public JacobianTraits<T>
{
public:
  JacobianBase(
    std::size_t M, std::size_t N, 
    const typename JacobianBase::NLSType &fn = nullptr
  ): _fn{fn}, _M{M}, _N{N} {
  };
  // initialize matrix?

  typename JacobianBase::JacType compute();

  typename JacobianBase::RealVec solve();

protected:
  const typename JacobianBase::NLSType _fn;
  std::size_t _M, _N;
  typename JacobianBase::JacType _J;

};


template <typename T>
class ForwardJac final : 
public JacobianBase<T>
{
  using dv = typename ForwardJac::dv;
  using FwArgType = typename ForwardJac::FwArgType;
  using FwRetType = typename ForwardJac::FwRetType;
  using NLSType = typename ForwardJac::NLSType;
  using JacType = typename ForwardJac::JacType;
  using RealVec = typename ForwardJac::RealVec;

public:
  ForwardJac(
    std::size_t M, std::size_t N, const NLSType &fn
  ): JacobianBase<T>(M, N, fn) {
    this->_J = JacType(M, N);
  };


  void compute(const RealVec &x0, RealVec &real_eval) {
    jacobian<T>(this->_fn, x0, real_eval, this->_J);
  }

  void compute_parallel(const RealVec &x0, RealVec &real_eval) {
    jacobian_parallel<T>(this->_fn, x0, real_eval, this->_J);
  }

  RealVec solve(const RealVec &x, RealVec &resid, int parallel=0) {

     // update the jacobian
    if (parallel) {
      compute_parallel(x, resid);
    } else {
      compute(x, resid);
    }
    return this->_J.fullPivLu().solve(resid);
  }

  JacType getJacobian() {
    return this->_J;
  }

};

#ifdef USE_CUDA

template <typename T>
using dv = typename JacobianTraits<T>::dv;

template <typename T>
using CudaArgType = typename JacobianTraits<T>::CudaArgType;
template <typename T>
using CudaRetType = typename JacobianTraits<T>::CudaRetType;
template <typename T>
using CudaDeviceFn = typename JacobianTraits<T>::CudaDeviceFn;

template <typename T>
using RealVec = typename JacobianTraits<T>::RealVec;
template <typename T>
using JacType = typename JacobianTraits<T>::JacType;

template <typename T>
class CudaJac final : 
public JacobianBase<T>
{
public:
  CudaJac(
    std::size_t M, std::size_t N, CudaFunctionWrapper<T> cuda_fn
  ): JacobianBase<T>(M, N, nullptr),
    _cuda_fn(cuda_fn)
  {
    this->_J = JacType<T>(M, N);
  };

  void compute(const RealVec<T> &x0) {
    RealVec<T> eval(M);
    jacobian_cuda<T>(_cuda_fn, x0, eval, this->_J)
  }

  RealVec<T> solve(const RealVec<T> &x, RealVec<T> &resid) {
    compute(x, resid);
    return this->_J.fullPivLu().solve(resid);
  }

  JacType<T> getJacobian() {
    return this->_J;
  }

  
protected:
  CudaFunctionWrapper<T> _cuda_fn;
};
#endif
}; // namespace forward
}; // namespace autodiff

#endif // JACOBIAN_HPP_