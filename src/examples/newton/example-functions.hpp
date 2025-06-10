#pragma once
#include <Eigen/Dense>
#include "../../autodiff/forward/CudaSupport.hpp"
#include "../../autodiff/forward/autodiff.hpp"
#include "Jacobian.hpp"

using dv   = autodiff::forward::DualVar<double>;
using dvec = Eigen::Matrix<dv, Eigen::Dynamic, 1>;

namespace testfun {

    constexpr int HEAVY_ITERS = 1;
    CUDA_HOST_DEVICE static inline dv heavy(const dv &v) {
        dv acc = 0;
        for(int i = 0; i < HEAVY_ITERS; ++i)
            acc = acc + sin(v)*exp(v) + cos(v)*tan(v);
        return acc;
    }

    // new helper: combine almost all of x via a shift
    CUDA_HOST_DEVICE static inline dv global_expr(const dvec &x, int shift) {
        dv acc = 0;
        int n = x.rows();
        for(int i = 0; i < n; ++i) {
            dv a = x[i];
            dv b = x[(i + shift) % n];
            // mix them up
            acc = acc + sin(a)*cos(b) + exp(a * b);
        }
        return acc;
    }

    // now define f0…f49 to use global_expr with different shifts
    #define MAKE_FN(Idx)                                  \
    CUDA_DEVICE dv f##Idx(const dvec &x) {                \
        return heavy(global_expr(x, (Idx)+1));            \
    }

    MAKE_FN(0) MAKE_FN(1) MAKE_FN(2) MAKE_FN(3) MAKE_FN(4) MAKE_FN(5) MAKE_FN(6) MAKE_FN(7) MAKE_FN(8) MAKE_FN(9)
    #undef MAKE_FN

    // test_fun0 can stay as-is—or you can call global_expr there too
    dvec test_fun0(const dvec &x) {
        dvec res(50);
        for(int i = 0; i < 50; ++i)
            res[i] = heavy(global_expr(x, i+1));
        return res;
    }
    #ifdef USE_CUDA
    // wrapper unchanged, but break the chained calls into separate statements
    newton::CudaFunctionWrapper<double> createcudafn0() {
      newton::CudaFunctionWrapper<double> cudafun(50);
    #define ADD(Idx) cudafun.add_output<f##Idx>();
      ADD(0) ADD(1) ADD(2) ADD(3) ADD(4) ADD(5) ADD(6) ADD(7) ADD(8) ADD(9)
    #undef ADD
        return cudafun;
    }
    #endif

} // namespace testfun
