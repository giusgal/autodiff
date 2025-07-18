#pragma once

#include <vector>
#include <functional>
#include <Eigen/Dense>

#include "DualVar.hpp"

namespace autodiff {
namespace forward {

template <typename T>
using DualVec = Eigen::Vector<autodiff::forward::DualVar<T>, Eigen::Dynamic>;
template <typename T>
using RealVec = Eigen::Vector<T, Eigen::Dynamic>;
template <typename T>
using JacType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using CudaDeviceFn = DualVar<T> (*)(const DualVec<T>&, const int out_idx);


template <typename T>
T derivative(std::function<DualVar<T>(DualVar<T>)> f, T x0){
    DualVar<T> res = f(DualVar<T>(x0, 1.0));
    return res.getInf();
}

template <typename T>
std::vector<T> gradient(
    std::function<DualVar<T>(std::vector<DualVar<T>>)> f, 
    std::vector<T> x
) {
    std::vector<DualVar<T>> xd;
    std::vector<T> res;

    xd.reserve(x.size());
    res.reserve(x.size());

    for(int i = 0; i < x.size(); i++){
        xd.push_back(DualVar<T>(x[i], 0.0));
    }

    for(int i = 0; i < x.size(); i++){
        xd[i].setInf(1.0);
        res.push_back(f(xd).getInf());
        xd[i].setInf(0.0);
    }

    return res;
}

template <typename T>
RealVec<T> gradient(
    std::function<DualVar<T>(DualVec<T>)> f,  
    RealVec<T> x
) {
    DualVec<T> xd(x.size());
    RealVec<T> res(x.size());

    for(int i = 0; i < x.size(); i++){
        xd[i] = DualVar<T>(x[i], 0.0);
    }

    for(int i = 0; i < x.size(); i++){
        xd[i].setInf(1.0);
        res[i] = f(xd).getInf();
        xd[i].setInf(0.0);
    }

    return res;
}



/**
 * Computes the jacobian of a function along with the value of that
 * function at the given point.
 * 
 * @param f Function whose jacobian is to be computed
 * @param x The point where the function and the jacobian must be evaluated
 * @param f_x (OUT) The value of the function at the given point
 * @param jac (OUT) The jacobian of the function at the given point
 */
template <typename T>
inline void jacobian(
    std::function<DualVec<T>(DualVec<T>)> f,
    RealVec<T> const & x,
    RealVec<T> & f_x,
    JacType<T> & jac
) {

    std::size_t input_dim = x.size();
    std::size_t output_dim;

    // create dual vector to feed the function as input
    DualVec<T> x0d(input_dim);
    DualVec<T> eval;

    for(int i = 0; i < input_dim; i++) {
      x0d[i] = DualVar<T>(x[i], 0.0);
    }

    eval = f(x0d);
    output_dim = eval.size();
    jac.resize(output_dim, input_dim);
    f_x.resize(output_dim);

    #pragma omp parallel for \
      firstprivate(x0d) lastprivate(eval) shared(jac)
    for (int i = 0; i < input_dim; i++) {
      x0d[i].setInf(1.0);
      eval = f(x0d);
      for (int j = 0; j < output_dim; j++) {
        jac(j, i) = eval[j].getInf();
      }
      x0d[i].setInf(0.0);
    }

    // write the value of fn in f_x pointer
    for (int i = 0; i < output_dim; i++) {
      f_x[i] = eval[i].getReal();
    }
}

#ifdef __CUDACC__

/**
 * @brief Registers a device function pointer with a specified CUDA device function.
 *
 * Assigns the provided device function pointer to the function to be registered. 
 * It is templated on the return type and the device function.
 *
 * @tparam T The underlying scalar type of the function
 * @tparam fn_to_be_registered The device function to be registered.
 * @param device_fn Pointer to the device function pointer to be set.
 */
template<typename T, CudaDeviceFn<T> fn_to_be_registered>
CUDA_GLOBAL \
void register_fn_device(CudaDeviceFn<T> *device_fn) {
    *device_fn = fn_to_be_registered;
}

/**
 * @class CudaFunctionWrapper
 * @brief A wrapper for passingCUDA device functions to the kernel.
 *
 * This class manages a pointer to a device function that computes a value and its derivative
 * using dual numbers. It provides methods to register a device function and to invoke it
 * from both host and device code.
 *
 * @tparam T The scalar type (e.g., float or double) for the computation.
 *
 */
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
  DualVar<T> operator()(const DualVec<T> &x, int y_i) const {
    return (*_device_fn)(x, y_i);
  }
  
};

/**
 * Cuda Kernel that calculates a function's jacobian and the function's evaluation
 * at a point x0.
 * 
 * @param input_dim The function's input dimension
 * @param output_dim The function's input dimension
 * @param x0 The point where the function and the jacobian must be evaluated
 * @param f_x (OUT) The value of the function at the given point
 * @param jac (OUT) The jacobian of the function at the given point
 * @param cuda_fn the wrapper containing the non linear system
 */
template <typename T>
CUDA_GLOBAL 
void jacobian_kernel_eval(
  std::size_t input_dim,
  std::size_t output_dim,
  const T *x0, 
  T *f_x,
  double *jac, 
  CudaFunctionWrapper<T> cuda_fn
) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x >= input_dim) return;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_y >= output_dim) return;

  // create local (dualvar) copy of input vector
  DualVec<T> x0_dual(input_dim);
  DualVar<T> y_dual;

  // prep local input
  for(int i = 0; i < input_dim; i++) {
    x0_dual[i] = DualVar<T>(x0[i]);
  }

  // seed the input
  x0_dual[tid_x].setInf(1.0);

  // evaluate function tid_y with seeded input
  y_dual = cuda_fn(x0_dual, tid_y);
  jac[tid_x * output_dim + tid_y] = y_dual.getInf();

  // save real function eval
  if (tid_x == 0) {
    f_x[tid_y] = y_dual.getReal();
  }
}


/**
 * Cuda Kernel that calculates a function's jacobian at a point x0.
 * Does not save the function evaluation for increased memory performance
 * and reduced control divergence.
 * 
 * @param input_dim The function's input dimension
 * @param output_dim The function's input dimension
 * @param x0 The point where the function and the jacobian must be evaluated
 * @param jac (OUT) The jacobian of the function at the given point
 * @param cuda_fn the wrapper containing the non linear system
 */
template <typename T>
CUDA_GLOBAL 
void jacobian_kernel(
  std::size_t input_dim,
  std::size_t output_dim,
  const T *x0, 
  double *jac, 
  CudaFunctionWrapper<T> cuda_fn
) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x >= input_dim) return;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_y >= output_dim) return;

  // create local (dualvar) copy of input vector
  DualVec<T> x0_dual(input_dim);
  DualVar<T> y_dual;

  // prep local input
  for(int i = 0; i < input_dim; i++) {
    x0_dual[i] = DualVar<T>(x0[i]);
  }

  // seed the input
  x0_dual[tid_x].setInf(1.0);

  // evaluate function tid_y with seeded input
  y_dual = cuda_fn(x0_dual, tid_y);
  jac[tid_x * output_dim + tid_y] = y_dual.getInf();
}

/**
 * Computes the jacobian of a function along with the value of that
 * function at the given point offloading function evalutation to a CUDA device.
 * 
 * @param f Function whose jacobian is to be computed
 * @param x The point where the function and the jacobian must be evaluated
 * @param f_x (OUT) The value of the function at the given point
 * @param jac (OUT) The jacobian of the function at the given point
 * @param eval if set to 1 will delegate the function evaluation to the device
 */
template <typename T>
void jacobian_cuda(
    CudaFunctionWrapper<T> f,
    RealVec<T> const & x,
    RealVec<T> & f_x,
    JacType<T> & jac,
    int eval = 0
) {
    std::size_t input_dim = x.size();
    std::size_t output_dim = f_x.size();

    double *jac_device;
    T *x0_device;
    T *f_x_device;
    CudaFunctionWrapper<T> *cudafn_device;

    // Reserve space on the device for the Jacobian
    CUDA_CHECK_ERROR(cudaMalloc(&jac_device, output_dim * input_dim * sizeof(T)));

    // Reserve space and copy input on the device
    CUDA_CHECK_ERROR(cudaMalloc(&x0_device, input_dim * sizeof(T)));
    CUDA_CHECK_ERROR(cudaMemcpy(x0_device, x.data(), input_dim * sizeof(T), cudaMemcpyHostToDevice));

    // Reserve space on the device for the real function evaluation
    if(eval)
        CUDA_CHECK_ERROR(cudaMalloc(&f_x_device, output_dim * sizeof(T)));

    // Reserve space on the device for the non linear system
    CUDA_CHECK_ERROR(cudaMalloc(&cudafn_device, output_dim * sizeof(CudaFunctionWrapper<T>)));

    // Since all threads which write in the same column of the jacobian are executing the same functions
    // place them in the same block
    dim3 blockDim;
    if (output_dim >= 256) {
        blockDim = dim3(256, 1);
    } else if (output_dim >= 128) {
        blockDim = dim3(128, 1);
    } else if (output_dim >= 32){
        blockDim = dim3(32, 1);   
    } else {
        blockDim = dim3(16, 1);
    }
    
    // Calculate grid dimensions to cover the entire matrix
    dim3 gridDim(
        (input_dim + blockDim.x - 1) / blockDim.x,
        (output_dim + blockDim.y - 1) / blockDim.y
    );

    // launch kernel and synchronize
    if (eval) {
        jacobian_kernel_eval<T><<<gridDim, blockDim>>>(input_dim, output_dim, x0_device, f_x_device, jac_device, f);
    } else {
        jacobian_kernel<T><<<gridDim, blockDim>>>(input_dim, output_dim, x0_device, jac_device, f);
    }
    CUDA_CHECK_ERROR(cudaGetLastError());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // copy real eval and jacobian onto host
    CUDA_CHECK_ERROR(cudaMemcpy(jac.data(), jac_device, output_dim * input_dim * sizeof(T), cudaMemcpyDeviceToHost));
    if(eval)
        CUDA_CHECK_ERROR(cudaMemcpy(f_x.data(), f_x_device, output_dim * sizeof(T), cudaMemcpyDeviceToHost));

    // free space
    CUDA_CHECK_ERROR(cudaFree(jac_device));
    if (eval)
        CUDA_CHECK_ERROR(cudaFree(f_x_device));
    
    
}


#endif // USE_CUDA

}; // namespace forward
}; // namespace autodiff
