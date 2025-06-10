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
        res[0] = x[0] * 1.0 + x[1] * 2.0 + x[2] * 4.0;
        res[1] = x[0] * 4.0 + x[1] * 5.0 + x[2] * 6.0;
        return res;
    }

    #ifdef USE_CUDA

    CUDA_DEVICE dv cu_f0(const dvec &x) { return x[0] * 1.0 + x[1] * 2.0 + x[2] * 4.0; };
    CUDA_DEVICE dv cu_f1(const dvec &x) { return x[0] * 4.0 + x[1] * 5.0 + x[2] * 6.0; };
    newton::CudaFunctionWrapper<double> createcudafn() {
        newton::CudaFunctionWrapper<double> cudafun(2);
        cudafun.add_output<cu_f0>();
        cudafun.add_output<cu_f1>();
        return cudafun;
    }

    #endif
} // namespace testfun
