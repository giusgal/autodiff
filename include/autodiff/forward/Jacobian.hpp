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

namespace autodiff {
namespace forward {

/**
 * @class JacobianBase
 * @brief Abstract base class for Jacobian matrix computation using automatic differentiation.
 * 
 * This class provides the common interface and data members for computing Jacobian matrices
 * of vector-valued functions f: R^N → R^M.
 * 
 * The class serves as a foundation for different implementation strategies (forward-mode AD,
 * CUDA-accelerated computation, etc.) and provides basic matrix storage and problem dimensions.
 * 
 * @tparam T The underlying scalar type for computations 
 */
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

/**
 * @class ForwardJac
 * @brief Forward-mode automatic differentiation implementation for Jacobian computation.
 * 
 * This class computes Jacobian matrices using forward-mode automatic differentiation with
 * dual numbers. For a function f: R^N → R^M, it performs N forward passes, each time
 * seeding one input variable with a unit perturbation (setting its infinitesimal part to 1)
 * while others remain at 0.
 * 
 * The computation strategy:
 *    For each input variable x_j (j = 0, ..., N-1):
 *    - Create dual number vector with x_j having infinitesimal part = 1
 *    - Evaluate f(x) to get dual result
 *    - Extract infinitesimal parts as column j of the Jacobian
 *
 * 
 * Features:
 * - Sequential and OpenMP parallel computation modes
 * - Integration with Eigen
 * - Support for Newton solving
 * 
 * Limitations:
 * - There is some wasted computation since the real part of the function evaluation 
 *   is discarded most of the time
 * 
 * @tparam T The underlying scalar type for computations 
 * 
 * @example
 * See test files for examples
 */
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


  void compute(const RealVec &x0, RealVec &real_eval = nullptr) {

    // create dual vector to feed the function as input
    FwArgType x0d(this->_N);
    FwRetType eval(this->_M);
    for(int i = 0; i < this->_N; i++) {
      x0d[i] = dv(x0[i], 0.0);
    }

    for (int i = 0; i < this->_N; i++) {
      x0d[i].setInf(1.0);
      eval = this->_fn(x0d);
      for (int j = 0; j < this->_M; j++) {
        this->_J(j, i) = eval[j].getInf();
      }
      x0d[i].setInf(0.0);
    }
    // write the value of fn in real_eval pointer
    if (real_eval != nullptr) {
      for (int i = 0; i < this->_M; i++) {
        real_eval[i] = eval[i].getReal();
      }
    }
  
  }

  void compute_parallel(const RealVec &x0, RealVec &real_eval) {

    // create dual vector to feed the function as input
    FwArgType x0d(this->_N);
    FwRetType eval(this->_M);
    for(int i = 0; i < this->_N; i++) {
      x0d[i] = dv(x0[i], 0.0);
    }

    #pragma omp parallel for \
      firstprivate(x0d) lastprivate(eval) shared(this->_J)
    for (int i = 0; i < this->_N; i++) {
      x0d[i].setInf(1.0);
      eval = this->_fn(x0d);
      for (int j = 0; j < this->_M; j++) {
        this->_J(j, i) = eval[j].getInf();
      }
      x0d[i].setInf(0.0);
    }
    // write the value of fn in real_eval pointer
    
    for (int i = 0; i < this->_M; i++) {
      real_eval[i] = eval[i].getReal();
    }
  
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


template<typename T, CudaDeviceFn<T> fn_to_be_registered>
CUDA_GLOBAL \
void register_fn_device(CudaDeviceFn<double> *device_fn) {
    *device_fn = fn_to_be_registered;
}


template <typename T>
CUDA_HOST_DEVICE \
struct CudaFunctionWrapper {
  
  CudaDeviceFn<T> *_device_fn;

  CudaFunctionWrapper() {
      CUDA_CHECK_ERROR(cudaMalloc(&_device_fn, sizeof(CudaDeviceFn<T>)));
    }

  template <CudaDeviceFn<T> device_fn>
  void register_fn_host() {
    register_fn_device<T, device_fn> <<<1, 1>>>(_device_fn);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  }


  CUDA_HOST_DEVICE \
  CudaRetType<T> operator()(const CudaArgType<T> &x, int y_i) const {
    return (*_device_fn)(x, y_i);
  }
  
};

template <typename T>
CUDA_GLOBAL 
void jacobian_kernel(
  int M, int N,
  const T *x0, 
  double *jac, 
  CudaFunctionWrapper<T> cuda_fn
) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x >= N) return;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_y >= M) return;

  // create local (dualvar) copy of input vector
  CudaArgType<T> x0_dual(N);
  CudaRetType<T> y_dual;

  // prep local input
  for(int i = 0; i < N; i++) {
    x0_dual[i] = dv<T>(x0[i]);
  }

  // seed the input
  x0_dual[tid_x].setInf(1.0);

  // evaluate function tid_y with seeded input
  y_dual = cuda_fn(x0_dual, tid_y);
  jac[tid_x * M + tid_y] = y_dual.getInf();

}

/**
 * @class CudaJac
 * @brief CUDA-accelerated Jacobian computation using forward-mode automatic differentiation.
 * 
 * This class leverages GPU parallelization to compute Jacobian matrices for large-scale problems.
 * Each GPU thread computes one element J(i,j) of the Jacobian matrix by evaluating the
 * function with appropriately seeded dual number inputs. Greater parallelism has been prioritized
 * over memory utilization.
 * 
 * Parallelization strategy:
 * - Each thread (tid_x, tid_y) computes J(tid_y, tid_x) = df_{tid_y}/dx_{tid_x}
 * - Threads are organized blocks to optimize memory access patterns. 
 * - All threads in a block evaluate the same function output to avoid control divergence.
 * 
 * Limitations:
 * - Functions must be expressed by the user in a fairly restrictive manner, explained below in more detail
 * - Memory overhead for large problems due to all threads requiring a copy of the input to seed
 * - There is quite a lot of wasted computation since all threads assigned to a specific output will 
 *   evaluate the real part of the same function at the same point.
 * 
 * @tparam T The underlying scalar type for computations (typically float or double)
 * 
 * @example
 * See test files for examples
 */
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
    int N = this->_N;
    int M = this->_M;

    double *jac_device;
    T *x0_device;
    CudaFunctionWrapper<T> *cudafn_device;

    // Allocate space for the jacobian on the device
    CUDA_CHECK_ERROR(cudaMalloc(&jac_device, M * N * sizeof(T)));

    // Allocate space and copy the input on the device
    CUDA_CHECK_ERROR(cudaMalloc(&x0_device, N * sizeof(T)));
    CUDA_CHECK_ERROR(cudaMemcpy(x0_device, x0.data(), N * sizeof(T), cudaMemcpyHostToDevice));

    // Allocate space for the function on the device
    CUDA_CHECK_ERROR(cudaMalloc(&cudafn_device, M * sizeof(CudaFunctionWrapper<T>)));

    // Since all threads which write in the same row of the jacobian are executing the same functions
    // place them in the same block
    dim3 blockDim;
    if (M >= 256) {
        blockDim = dim3(256, 1);
    } else if (M >= 128) {
        blockDim = dim3(128, 1);
    } else if (M >= 32){
        blockDim = dim3(32, 1);   
    } else {
        blockDim = dim3(16, 1);
    }
    
    // Calculate grid dimensions to cover the entire matrix
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );
    
    // Launch kernel
    jacobian_kernel<T><<<gridDim, blockDim>>>(M, N, x0_device, jac_device, _cuda_fn);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Retrieve output and free device memory
    CUDA_CHECK_ERROR(cudaMemcpy(this->_J.data(), jac_device, M * N * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(jac_device));
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