#pragma once
#include <Eigen/Dense>
#include "../../autodiff/forward/CudaSupport.hpp"
#include "../../autodiff/forward/autodiff.hpp"
#include "Jacobian.hpp"

using dv = autodiff::forward::DualVar<double>;
using dvec = Eigen::Matrix<dv, Eigen::Dynamic, 1>;

namespace testfun {
    dvec test_fun(const dvec &x) {
        dvec res(2);
        res[0] = x[0] * 3.0;
        res[1] = x[0] * 2.0;
        return res;
    }

    #ifdef USE_CUDA
    CUDA_DEVICE dv cudafun(const dvec &x, const int out_idx) {
        switch (out_idx)
        {
        case 0:
            return x[0] * 3.0;
            break;
        case 1:
            return x[0] * 2.0;
            break;
        default:
            break;
        }
        return 0;
    }
    
    newton::CudaFunctionWrapper<double> createwrapper() {
        newton::CudaFunctionWrapper<double> cu_wrapper<cudafun>();
        return cu_wrapper;
    }
    #endif
} // namespace testfun