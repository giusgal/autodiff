#pragma once
#include <functional>
#include <Eigen/Dense>
#include "NewtonTraits.hpp"
#include <omp.h>
#include "../../autodiff/forward/CudaSupport.hpp"
#ifdef USE_CUDA
#define CUDA_LINSYS_MAXOUT 10
#include <cuda_runtime.h>


#endif


namespace newton 
{

template <typename T>
class JacobianBase : public NewtonTraits<T>
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

    // create dual vector to feed the function as input
    FwArgType x0d(this->_N);
    FwRetType eval(this->_N);
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
    
    for (int i = 0; i < this->_M; i++) {
      real_eval[i] = eval[i].getReal();
    }
  
  }

  void compute_parallel(const RealVec &x0, RealVec &real_eval) {
    // create dual vector to feed the function as input
    FwArgType x0d(this->_N);
    FwRetType eval(this->_N);
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
using RealVec = typename NewtonTraits<T>::RealVec;
template <typename T>
using dv = typename NewtonTraits<T>::dv;
template <typename T>
using CudaArgType = typename NewtonTraits<T>::FwArgType;
template <typename T>
using CudaNLSType = typename NewtonTraits<T>::CudaNLSType;
template <typename T>
using CudaRetType = dv<T>;
template <typename T>
using JacType = typename NewtonTraits<T>::JacType;
template <typename T>
using CudaDeviceFn = typename NewtonTraits<T>::CudaDeviceFn;
template <typename T, CudaDeviceFn<T> F>
using SetupKernelFn = typename NewtonTraits<T>::SetupKernelFn<F>;
template <typename T, CudaDeviceFn<T> F>
using SetupKernelFn = typename NewtonTraits<T>::template SetupKernelFn<F>;


////THREADS IN THE SAME BLOCK SHOULD EXECUTE THE SAME FUNCTION

template<typename T, CudaDeviceFn<T> fn_to_be_registered>
CUDA_GLOBAL \
void register_fn(CudaDeviceFn<double> *device_fn_array, int idx) {
    device_fn_array[idx] = fn_to_be_registered;
}

template <typename T>
CUDA_HOST_DEVICE \
struct CudaFunctionWrapper {
  
  CudaDeviceFn<T> *_device_fns;
  int _out_dim, _out_max_dim;

  CudaFunctionWrapper(int out_max_dim): 
    _out_dim(0), _out_max_dim(out_max_dim) {
      CUDA_CHECK_ERROR(cudaMalloc(&_device_fns, _out_max_dim * sizeof(CudaDeviceFn<T>)));
    }

  template <CudaDeviceFn<T> device_fn>
  void add_output() {
    if (_out_dim >= _out_max_dim) {
      std::cout << "Attempting to add functions over the maximum size" << std::endl;
      exit(0);
    }
    register_fn<T, device_fn> <<<1, 1>>>(_device_fns, _out_dim);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    _out_dim ++;
  }

  CUDA_HOST_DEVICE \
  CudaRetType<T> operator()(const CudaArgType<T> &x, int y_i) const {
    return _device_fns[y_i](x);
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
  dv<T> y_dual;

  // prep local input
  for(int i = 0; i < N; i++) {
    x0_dual[i] = dv<T>(x0[i]);
  }

  // seed the input
  x0_dual[tid_x].setInf(1.0);

  // evaluate function tid_y with seeded input
  y_dual = cuda_fn(x0_dual, tid_y);

  jac[tid_x + tid_y * N] = y_dual.getInf();

}

#endif

#ifdef USE_CUDA





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

    CUDA_CHECK_ERROR(cudaMalloc(&jac_device, M * N * sizeof(T)));

    CUDA_CHECK_ERROR(cudaMalloc(&x0_device, N * sizeof(T)));
    CUDA_CHECK_ERROR(cudaMemcpy(x0_device, x0.data(), N * sizeof(T), cudaMemcpyHostToDevice));
    std::cout << "Size of wrapper: " << M * sizeof(CudaFunctionWrapper<T>) << std::endl;
    CUDA_CHECK_ERROR(cudaMalloc(&cudafn_device, M * sizeof(CudaFunctionWrapper<T>)));

    // Since all threads which write in the same column of the jacobian are executing the same functions
    // place them in the same block
    // cap threads per block at 256?
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
    jacobian_kernel<T><<<gridDim, blockDim>>>(M, N, x0_device, jac_device, _cuda_fn);
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaMemcpy(this->_J.data(), jac_device, M * N * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaFree(jac_device));
  }

  JacType<T> getJacobian() {
    return this->_J;
  }
protected:
  CudaFunctionWrapper<T> _cuda_fn;
};
#else
#endif



// template <typename T>
// class ManualJac : public JacobianBase<T>
// {
// public:
//   ManualJac(
//     fn_type<T> &_jacfn, _M, _N
//   ):
//   jacfn{_jacfn}, M{_M}, N{_N} {};

//   Eigen::MatrixXd compute(fn_io_type<T> &x0) {
//     return jacfn(x0);
//   }


// };


//TODO

// template <typename T>
// class ReverseJac : public JacobianBase<Var<T>> {

// }


}// namespace newton