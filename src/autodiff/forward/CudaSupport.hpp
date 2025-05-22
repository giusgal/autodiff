#pragma once

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