#pragma once

#include <functional>
#include <Eigen/Dense>
#include <omp.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "JacobianTraits.hpp"
#include "ForwardEigenSupport.hpp"
#include "CudaSupport.hpp"
#include "ForwardUtility.hpp"

#include "ReverseEigenSupport.hpp"
#include "ReverseUtility.hpp"

namespace newton {

/**
 * @class JacobianBase
 * @brief An abstract class which represents a generic jacobian
 */
class JacobianBase : public JacobianTraits {
public:
    JacobianBase(std::size_t M, std::size_t N): _M{M}, _N{N}, _J{M,N} {}

    virtual ~JacobianBase() = default;

    virtual void compute(const RealVec &, RealVec &) = 0;
    virtual RealVec solve(const RealVec &, RealVec &) = 0;

    JacType getJacobian() const {
        return _J;
    }
protected:
    size_t _M;
    size_t _N;
    JacType _J;
};


/**
 * @class ForwardJacobian 
 * @brief ForwardJacobian allows to solve systems involving
 *  the jacobian of a given non-linear function using 
 *  forward-mode automatic differtentiation
 */
class ForwardJac final : public JacobianBase {
public:
    ForwardJac(std::size_t M, std::size_t N, FwNLSType const & fn):
        JacobianBase(M, N), _fn{fn} {}

    void compute(const RealVec & x0, RealVec & real_eval) override {
        jacobian<double>(_fn, x0, real_eval, this->_J);
    }

    RealVec solve(const RealVec & x, RealVec & resid) override {
        // update the jacobian
        compute(x, resid);
        return this->_J.fullPivLu().solve(resid);
    }
protected:
    FwNLSType const & _fn;
};

/**
 * @class ReverseJac
 * @brief ReverseJac allows to solve systems involving
 *  the jacobian of a given non-linear function using
 *  reverse-mode automatic differtentiation
 */
class ReverseJac final : public JacobianBase {
public:
    ReverseJac(std::size_t M, std::size_t N, RvNLSType const & fn):
      JacobianBase(M, N), _fn{fn} {}
  
    void compute(const RealVec & x0, RealVec & real_eval) override {
        jacobian(_fn, x0, real_eval, this->_J);
    }

    RealVec solve(const RealVec & x, RealVec & resid) override {
        // update the jacobian
        compute(x, resid);
        return this->_J.fullPivLu().solve(resid);
    }
protected:
    RvNLSType const & _fn;
};

#ifdef __CUDACC__

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
#endif // __CUDACC__
}; // namespace newton
