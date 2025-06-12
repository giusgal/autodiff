#pragma once
#include <Eigen/Dense>
#include "../../autodiff/forward/CudaSupport.hpp"
#include "../../autodiff/forward/autodiff.hpp"
#include "Jacobian.hpp"

using dv = autodiff::forward::DualVar<double>;
using dvec = Eigen::Matrix<dv, Eigen::Dynamic, 1>;

namespace testfun {
    dvec test_fun(const dvec &x) {
        dvec res(1);
        res[0] = x[0] * 1.0 + x[1] * 2.0 + x[2] * 4.0;
        return res;
    }

    #ifdef USE_CUDA

    CUDA_DEVICE dv cu_f0(const dvec &x, const int y) { return x[0] * 1.0 + x[1] * 2.0 + x[2] * 4.0; };
    newton::CudaFunctionWrapper<double> createcudafn() {
        newton::CudaFunctionWrapper<double> cudafun;
        cudafun.add_output<cu_f0>();
        return cudafun;
    }
    #endif
} // namespace testfun