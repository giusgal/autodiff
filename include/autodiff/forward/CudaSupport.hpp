#ifndef CUDA_SUPPORT_HPP_
#define CUDA_SUPPORT_HPP_

/* 
  Allows annotating all functions that need to be available within the kernel with aliases which are 
  substituted with the proper __device__ (or similar) annotation if the compiler flag -DUSE_CUDA is used.
  If the flag is not present, the aliases don't do anything.
*/

#ifdef USE_CUDA

#include <cuda_runtime.h>

#define CUDA_HOST_DEVICE __host__ __device__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__
#define CUDA_CHECK_ERROR(err) \
  do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
    } \
  } while (0)


#else

#define CUDA_HOST_DEVICE
#define CUDA_DEVICE
#define CUDA_GLOBAL
#define CUDA_CHECK_ERROR(err) ((void)0)


#endif // USE_CUDA

#endif //CUDA_SUPPORT_HPP_