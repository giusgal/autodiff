#pragma once
#include <Eigen/Dense>
#include "../../autodiff/forward/CudaSupport.hpp"
#include "../../autodiff/forward/autodiff.hpp"
#include "Jacobian.hpp"

using dv = autodiff::forward::DualVar<double>;
using dvec = Eigen::Matrix<dv, Eigen::Dynamic, 1>;

namespace testfun {
    dvec test_fun0(const dvec &x) {
        dvec res(50);
        res[0] = exp(x[39] * 0.1) + tan(x[10] * 0.1); // f0
        res[1] = log(1.0 + x[18] * x[18]) + x[15]; // f1
        res[2] = log(1.0 + x[34] * x[34]) + x[32]; // f2
        res[3] = exp(x[5] * 0.1) + tan(x[41] * 0.1) - x[15]; // f3
        res[4] = exp(x[23]) - x[19]; // f4
        res[5] = log(1.0 + x[1] * x[1]) + x[37]; // f5
        res[6] = exp(x[35] * 0.1) + tan(x[26] * 0.1); // f6
        res[7] = exp(x[23]) - x[46]; // f7
        res[8] = sin(x[25]) + x[41] * x[41]; // f8
        res[9] = sin(x[12]) + x[4] * x[4]; // f9
        res[10] = x[0] * x[46] + cos(x[9]); // f10
        res[11] = sin(x[42]) + x[20] * x[20]; // f11
        res[12] = sin(x[43]) + x[7] * x[7]; // f12
        res[13] = exp(x[23]) - x[8]; // f13
        res[14] = exp(x[17]) - x[26]; // f14
        res[15] = x[45] * x[40] + cos(x[30]); // f15
        res[16] = exp(x[32]) - x[1]; // f16
        res[17] = x[31] * x[34] + cos(x[19]); // f17
        res[18] = exp(x[34] * 0.1) + tan(x[7] * 0.1) - x[47]; // f18
        res[19] = exp(x[8] * 0.1) + tan(x[38] * 0.1) - x[36]; // f19
        res[20] = log(1.0 + x[8] * x[8]) + x[42]; // f20
        res[21] = exp(x[12] * 0.1) + tan(x[33] * 0.1) - x[44]; // f21
        res[22] = sin(x[27]) + x[7] * x[7]; // f22
        res[23] = sin(x[45]) + x[6] * x[6]; // f23
        res[24] = exp(x[31] * 0.1) + tan(x[1] * 0.1) - x[43]; // f24
        res[25] = x[10] * x[45] + cos(x[44]); // f25
        res[26] = x[33] * x[28] + cos(x[19]); // f26
        res[27] = exp(x[10]) - x[23]; // f27
        res[28] = exp(x[12] * 0.1) + tan(x[39] * 0.1); // f28
        res[29] = exp(x[23] * 0.1) + tan(x[7] * 0.1); // f29
        res[30] = exp(x[18]) - x[14]; // f30
        res[31] = x[20] * x[39] + cos(x[9]); // f31
        res[32] = exp(x[16]) - x[40]; // f32
        res[33] = exp(x[4] * 0.1) + tan(x[29] * 0.1); // f33
        res[34] = exp(x[32] * 0.1) + tan(x[25] * 0.1) - x[40]; // f34
        res[35] = x[35] * x[49] + cos(x[4]); // f35
        res[36] = x[14] * x[44] + cos(x[38]); // f36
        res[37] = log(1.0 + x[41] * x[41]) + x[42]; // f37
        res[38] = x[34] * x[5] + cos(x[29]); // f38
        res[39] = sin(x[16]) + x[9] * x[9]; // f39
        res[40] = exp(x[18]) - x[23]; // f40
        res[41] = exp(x[46] * 0.1) + tan(x[10] * 0.1); // f41
        res[42] = exp(x[30] * 0.1) + tan(x[25] * 0.1); // f42
        res[43] = exp(x[42]) - x[23]; // f43
        res[44] = exp(x[13] * 0.1) + tan(x[0] * 0.1); // f44
        res[45] = exp(x[11]) - x[27]; // f45
        res[46] = exp(x[28] * 0.1) + tan(x[10] * 0.1) - x[36]; // f46
        res[47] = log(1.0 + x[48] * x[48]) + x[3]; // f47
        res[48] = x[5] * x[46] + cos(x[16]); // f48
        res[49] = exp(x[21]) - x[24]; // f49
        return res;
    }

    CUDA_DEVICE dv f0(const dvec &x) { return exp(x[39] * 0.1) + tan(x[10] * 0.1); }
    CUDA_DEVICE dv f1(const dvec &x) { return log(1.0 + x[18] * x[18]) + x[15]; }
    CUDA_DEVICE dv f2(const dvec &x) { return log(1.0 + x[34] * x[34]) + x[32]; }
    CUDA_DEVICE dv f3(const dvec &x) { return exp(x[5] * 0.1) + tan(x[41] * 0.1) - x[15]; }
    CUDA_DEVICE dv f4(const dvec &x) { return exp(x[23]) - x[19]; }
    CUDA_DEVICE dv f5(const dvec &x) { return log(1.0 + x[1] * x[1]) + x[37]; }
    CUDA_DEVICE dv f6(const dvec &x) { return exp(x[35] * 0.1) + tan(x[26] * 0.1); }
    CUDA_DEVICE dv f7(const dvec &x) { return exp(x[23]) - x[46]; }
    CUDA_DEVICE dv f8(const dvec &x) { return sin(x[25]) + x[41] * x[41]; }
    CUDA_DEVICE dv f9(const dvec &x) { return sin(x[12]) + x[4] * x[4]; }
    CUDA_DEVICE dv f10(const dvec &x) { return x[0] * x[46] + cos(x[9]); }
    CUDA_DEVICE dv f11(const dvec &x) { return sin(x[42]) + x[20] * x[20]; }
    CUDA_DEVICE dv f12(const dvec &x) { return sin(x[43]) + x[7] * x[7]; }
    CUDA_DEVICE dv f13(const dvec &x) { return exp(x[23]) - x[8]; }
    CUDA_DEVICE dv f14(const dvec &x) { return exp(x[17]) - x[26]; }
    CUDA_DEVICE dv f15(const dvec &x) { return x[45] * x[40] + cos(x[30]); }
    CUDA_DEVICE dv f16(const dvec &x) { return exp(x[32]) - x[1]; }
    CUDA_DEVICE dv f17(const dvec &x) { return x[31] * x[34] + cos(x[19]); }
    CUDA_DEVICE dv f18(const dvec &x) { return exp(x[34] * 0.1) + tan(x[7] * 0.1) - x[47]; }
    CUDA_DEVICE dv f19(const dvec &x) { return exp(x[8] * 0.1) + tan(x[38] * 0.1) - x[36]; }
    CUDA_DEVICE dv f20(const dvec &x) { return log(1.0 + x[8] * x[8]) + x[42]; }
    CUDA_DEVICE dv f21(const dvec &x) { return exp(x[12] * 0.1) + tan(x[33] * 0.1) - x[44]; }
    CUDA_DEVICE dv f22(const dvec &x) { return sin(x[27]) + x[7] * x[7]; }
    CUDA_DEVICE dv f23(const dvec &x) { return sin(x[45]) + x[6] * x[6]; }
    CUDA_DEVICE dv f24(const dvec &x) { return exp(x[31] * 0.1) + tan(x[1] * 0.1) - x[43]; }
    CUDA_DEVICE dv f25(const dvec &x) { return x[10] * x[45] + cos(x[44]); }
    CUDA_DEVICE dv f26(const dvec &x) { return x[33] * x[28] + cos(x[19]); }
    CUDA_DEVICE dv f27(const dvec &x) { return exp(x[10]) - x[23]; }
    CUDA_DEVICE dv f28(const dvec &x) { return exp(x[12] * 0.1) + tan(x[39] * 0.1); }
    CUDA_DEVICE dv f29(const dvec &x) { return exp(x[23] * 0.1) + tan(x[7] * 0.1); }
    CUDA_DEVICE dv f30(const dvec &x) { return exp(x[18]) - x[14]; }
    CUDA_DEVICE dv f31(const dvec &x) { return x[20] * x[39] + cos(x[9]); }
    CUDA_DEVICE dv f32(const dvec &x) { return exp(x[16]) - x[40]; }
    CUDA_DEVICE dv f33(const dvec &x) { return exp(x[4] * 0.1) + tan(x[29] * 0.1); }
    CUDA_DEVICE dv f34(const dvec &x) { return exp(x[32] * 0.1) + tan(x[25] * 0.1) - x[40]; }
    CUDA_DEVICE dv f35(const dvec &x) { return x[35] * x[49] + cos(x[4]); }
    CUDA_DEVICE dv f36(const dvec &x) { return x[14] * x[44] + cos(x[38]); }
    CUDA_DEVICE dv f37(const dvec &x) { return log(1.0 + x[41] * x[41]) + x[42]; }
    CUDA_DEVICE dv f38(const dvec &x) { return x[34] * x[5] + cos(x[29]); }
    CUDA_DEVICE dv f39(const dvec &x) { return sin(x[16]) + x[9] * x[9]; }
    CUDA_DEVICE dv f40(const dvec &x) { return exp(x[18]) - x[23]; }
    CUDA_DEVICE dv f41(const dvec &x) { return exp(x[46] * 0.1) + tan(x[10] * 0.1); }
    CUDA_DEVICE dv f42(const dvec &x) { return exp(x[30] * 0.1) + tan(x[25] * 0.1); }
    CUDA_DEVICE dv f43(const dvec &x) { return exp(x[42]) - x[23]; }
    CUDA_DEVICE dv f44(const dvec &x) { return exp(x[13] * 0.1) + tan(x[0] * 0.1); }
    CUDA_DEVICE dv f45(const dvec &x) { return exp(x[11]) - x[27]; }
    CUDA_DEVICE dv f46(const dvec &x) { return exp(x[28] * 0.1) + tan(x[10] * 0.1) - x[36]; }
    CUDA_DEVICE dv f47(const dvec &x) { return log(1.0 + x[48] * x[48]) + x[3]; }
    CUDA_DEVICE dv f48(const dvec &x) { return x[5] * x[46] + cos(x[16]); }
    CUDA_DEVICE dv f49(const dvec &x) { return exp(x[21]) - x[24]; }

    newton::CudaFunctionWrapper<double> createcudafn0() {
        newton::CudaFunctionWrapper<double> cudafun(50);
        cudafun.add_output<f0>();
        cudafun.add_output<f1>();
        cudafun.add_output<f2>();
        cudafun.add_output<f3>();
        cudafun.add_output<f4>();
        cudafun.add_output<f5>();
        cudafun.add_output<f6>();
        cudafun.add_output<f7>();
        cudafun.add_output<f8>();
        cudafun.add_output<f9>();
        cudafun.add_output<f10>();
        cudafun.add_output<f11>();
        cudafun.add_output<f12>();
        cudafun.add_output<f13>();
        cudafun.add_output<f14>();
        cudafun.add_output<f15>();
        cudafun.add_output<f16>();
        cudafun.add_output<f17>();
        cudafun.add_output<f18>();
        cudafun.add_output<f19>();
        cudafun.add_output<f20>();
        cudafun.add_output<f21>();
        cudafun.add_output<f22>();
        cudafun.add_output<f23>();
        cudafun.add_output<f24>();
        cudafun.add_output<f25>();
        cudafun.add_output<f26>();
        cudafun.add_output<f27>();
        cudafun.add_output<f28>();
        cudafun.add_output<f29>();
        cudafun.add_output<f30>();
        cudafun.add_output<f31>();
        cudafun.add_output<f32>();
        cudafun.add_output<f33>();
        cudafun.add_output<f34>();
        cudafun.add_output<f35>();
        cudafun.add_output<f36>();
        cudafun.add_output<f37>();
        cudafun.add_output<f38>();
        cudafun.add_output<f39>();
        cudafun.add_output<f40>();
        cudafun.add_output<f41>();
        cudafun.add_output<f42>();
        cudafun.add_output<f43>();
        cudafun.add_output<f44>();
        cudafun.add_output<f45>();
        cudafun.add_output<f46>();
        cudafun.add_output<f47>();
        cudafun.add_output<f48>();
        cudafun.add_output<f49>();
        return cudafun;
    }

} // namespace testfun
