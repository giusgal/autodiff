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

namespace newton {


class JacobianBase : public JacobianTraits
{
public:
  JacobianBase(
    std::size_t M, std::size_t N
  ): _M{M}, _N{N} {
  };
  // initialize matrix?

  virtual ~JacobianBase() = default;

  virtual void compute(const RealVec &, RealVec &) = 0;

  virtual RealVec solve(const RealVec &, RealVec &) = 0;

protected:
  std::size_t _M, _N;
  JacType _J;
};


class ForwardJac final : 
public JacobianBase
{
  // using dv = typename ForwardJac::dv;
  // using FwArgType = typename ForwardJac::FwArgType;
  // using FwRetType = typename ForwardJac::FwRetType;
  // using FwNLSType = typename ForwardJac::fwNLSType;
  // using JacType = typename ForwardJac::JacType;
  // using RealVec = typename ForwardJac::RealVec;

public:
  ForwardJac(
    std::size_t M, std::size_t N, const FwNLSType &fn
  ): JacobianBase(M, N) {
    this->_J = JacType(M, N);
    this->_fn = fn;
  };
  
  void compute(const RealVec &x0, RealVec &real_eval) override {
    jacobian<double>(this->_fn, x0, real_eval, this->_J);
  }

  RealVec solve(const RealVec &x, RealVec &resid) override {
    // update the jacobian
    compute(x, resid);
    return this->_J.fullPivLu().solve(resid);
  }

  JacType getJacobian() {
    return this->_J;
  }
protected:
  FwNLSType _fn;
};

#ifdef USE_CUDA

// template <typename T>
// using dv = typename JacobianTraits<T>::dv;

// template <typename T>
// using CudaArgType = typename JacobianTraits<T>::CudaArgType;
// template <typename T>
// using CudaRetType = typename JacobianTraits<T>::CudaRetType;
// template <typename T>
// using CudaDeviceFn = typename JacobianTraits<T>::CudaDeviceFn;

// template <typename T>
// using RealVec = typename JacobianTraits<T>::RealVec;
// template <typename T>
// using JacType = typename JacobianTraits<T>::JacType;


class CudaJac final : 
public JacobianBase
{
public:
  CudaJac(
    std::size_t M, std::size_t N, CudaFunctionWrapper<double> cuda_fn
  ): JacobianBase(M, N),
    _cuda_fn(cuda_fn)
  {
    this->_J = JacType(M, N);
  };

  void compute(const RealVec &x0) {
    RealVec eval(M);
    jacobian_cuda<double>(_cuda_fn, x0, eval, this->_J)
  }

  RealVec solve(const RealVec &x, RealVec &resid) {
    compute(x, resid);
    return this->_J.fullPivLu().solve(resid);
  }

  JacType getJacobian() {
    return this->_J;
  }

  
protected:
  CudaFunctionWrapper<double> _cuda_fn;
};
#endif
}; // namespace newton

#endif // JACOBIAN_HPP_