#ifndef EXAMPLE_FUNCTIONS_HPP_
#define EXAMPLE_FUNCTIONS_HPP_

#include <Eigen/Dense>
#include <cmath>
#include "DualVar.hpp"
#include "CudaSupport.hpp"

using dv = autodiff::forward::DualVar<double>;
using dvec = Eigen::Matrix<dv, Eigen::Dynamic, 1>;

namespace testfun {
    constexpr int input_dim = 10;
    constexpr int output_dim = 10;
    constexpr int complexity = 5;

    dvec test_fun(const dvec &x) {
        dvec res(10);
        dv acc;
        
        // Function output 0 (depends on variables: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        acc = 0;
        for(int j = 0; j < complexity; j++){
            acc = acc + (abs(x[2]) * 0.273 + (x[7] + x[6]) * 0.224 + (x[0] + x[6]) * 0.275 + sin(x[2]) * 0.212 + abs(x[0]) * 0.292) / static_cast<double>(complexity);
        }
        res[0] = acc;
        
        // Function output 1 (depends on variables: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        acc = 0;
        for(int j = 0; j < complexity; j++){
            acc = acc + ((x[7] - x[0]) * 0.259 + (x[3] * x[7]) * 0.0201 + (x[9] * x[1]) * 0.0129 + (x[7] * x[2]) * 0.0492 + (x[9] * x[1]) * 0.0132) / static_cast<double>(complexity);
        }
        res[1] = acc;
        
        // Function output 2 (depends on variables: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        acc = 0;
        for(int j = 0; j < complexity; j++){
            acc = acc + ((x[4] + x[3]) * 0.180 + (x[1] * x[9]) * 0.0029 + sqrt(x[0] * x[0] + 0.01) * 0.198 + (x[6] * x[6]) * 0.0053 + cos(x[4]) * 0.280) / static_cast<double>(complexity);
        }
        res[2] = acc;
        
        // Function output 3 (depends on variables: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        acc = 0;
        for(int j = 0; j < complexity; j++){
            acc = acc + ((x[9] * x[6]) * 0.0122 + abs(x[8]) * 0.170 + (x[0] + x[9]) * 0.285 + (x[1] - x[7]) * 0.137 + (x[1] * x[8] * x[4]) * 0.0232) / static_cast<double>(complexity);
        }
        res[3] = acc;
        
        // Function output 4 (depends on variables: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        acc = 0;
        for(int j = 0; j < complexity; j++){
            acc = acc + (abs(x[8]) * 0.239 + (x[6] + x[4]) * 0.249 + sqrt(x[4] * x[4] + 0.01) * 0.067 + sqrt(x[3] * x[3] + 0.01) * 0.288 + (x[4] * x[5] * x[9]) * 0.0193) / static_cast<double>(complexity);
        }
        res[4] = acc;
        
        // Function output 5 (depends on variables: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        acc = 0;
        for(int j = 0; j < complexity; j++){
            acc = acc + ((x[5] * x[9]) * 0.0636 + cos(x[1]) * 0.149 + exp(-x[4] * x[4] * 0.1) * 0.161 + (x[8] * x[3] * x[7]) * 0.0118 + (x[9] * x[6]) * 0.0537) / static_cast<double>(complexity);
        }
        res[5] = acc;
        
        // Function output 6 (depends on variables: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        acc = 0;
        for(int j = 0; j < complexity; j++){
            acc = acc + ((x[4] * x[2]) * 0.0057 + log(x[6] * x[6] + 1.0) * 0.198 + log(x[7] * x[7] + 1.0) * 0.093 + (x[2] * x[1]) * 0.0342 + log(x[2] * x[2] + 1.0) * 0.198) / static_cast<double>(complexity);
        }
        res[6] = acc;
        
        // Function output 7 (depends on variables: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        acc = 0;
        for(int j = 0; j < complexity; j++){
            acc = acc + (cos(x[2]) * 0.264 + (x[3] * x[0]) * 0.0258 + (x[0] * x[9]) * 0.0340 + (x[9] * x[2] * x[1]) * 0.0257 + cos(x[6]) * 0.092) / static_cast<double>(complexity);
        }
        res[7] = acc;
        
        // Function output 8 (depends on variables: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        acc = 0;
        for(int j = 0; j < complexity; j++){
            acc = acc + (sqrt(x[6] * x[6] + 0.01) * 0.267 + (x[4] * x[9] * x[3]) * 0.0145 + (x[3] + x[5]) * 0.257 + (x[3] * x[0]) * 0.0123 + (x[3] * x[0]) * 0.0053) / static_cast<double>(complexity);
        }
        res[8] = acc;
        
        // Function output 9 (depends on variables: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        acc = 0;
        for(int j = 0; j < complexity; j++){
            acc = acc + (sqrt(x[1] * x[1] + 0.01) * 0.203 + sin(x[0]) * 0.079 + cos(x[2]) * 0.246 + (x[5] * x[4] * x[9]) * 0.0102 + cos(x[4]) * 0.159) / static_cast<double>(complexity);
        }
        res[9] = acc;
        
        return res;
    }

    #ifdef USE_CUDA

    CUDA_DEVICE dv cu_f0(const dvec &x, const int y) { 
        dv acc = 0;
        switch (y)
        {
        case 0:
            for(int j = 0; j < complexity; j++){
                acc = acc + (abs(x[2]) * 0.273 + (x[7] + x[6]) * 0.224 + (x[0] + x[6]) * 0.275 + sin(x[2]) * 0.212 + abs(x[0]) * 0.292) / static_cast<double>(complexity);
            }
            break;
        case 1:
            for(int j = 0; j < complexity; j++){
                acc = acc + ((x[7] - x[0]) * 0.259 + (x[3] * x[7]) * 0.0201 + (x[9] * x[1]) * 0.0129 + (x[7] * x[2]) * 0.0492 + (x[9] * x[1]) * 0.0132) / static_cast<double>(complexity);
            }
            break;
        case 2:
            for(int j = 0; j < complexity; j++){
                acc = acc + ((x[4] + x[3]) * 0.180 + (x[1] * x[9]) * 0.0029 + sqrt(x[0] * x[0] + 0.01) * 0.198 + (x[6] * x[6]) * 0.0053 + cos(x[4]) * 0.280) / static_cast<double>(complexity);
            }
            break;
        case 3:
            for(int j = 0; j < complexity; j++){
                acc = acc + ((x[9] * x[6]) * 0.0122 + abs(x[8]) * 0.170 + (x[0] + x[9]) * 0.285 + (x[1] - x[7]) * 0.137 + (x[1] * x[8] * x[4]) * 0.0232) / static_cast<double>(complexity);
            }
            break;
        case 4:
            for(int j = 0; j < complexity; j++){
                acc = acc + (abs(x[8]) * 0.239 + (x[6] + x[4]) * 0.249 + sqrt(x[4] * x[4] + 0.01) * 0.067 + sqrt(x[3] * x[3] + 0.01) * 0.288 + (x[4] * x[5] * x[9]) * 0.0193) / static_cast<double>(complexity);
            }
            break;
        case 5:
            for(int j = 0; j < complexity; j++){
                acc = acc + ((x[5] * x[9]) * 0.0636 + cos(x[1]) * 0.149 + exp(-x[4] * x[4] * 0.1) * 0.161 + (x[8] * x[3] * x[7]) * 0.0118 + (x[9] * x[6]) * 0.0537) / static_cast<double>(complexity);
            }
            break;
        case 6:
            for(int j = 0; j < complexity; j++){
                acc = acc + ((x[4] * x[2]) * 0.0057 + log(x[6] * x[6] + 1.0) * 0.198 + log(x[7] * x[7] + 1.0) * 0.093 + (x[2] * x[1]) * 0.0342 + log(x[2] * x[2] + 1.0) * 0.198) / static_cast<double>(complexity);
            }
            break;
        case 7:
            for(int j = 0; j < complexity; j++){
                acc = acc + (cos(x[2]) * 0.264 + (x[3] * x[0]) * 0.0258 + (x[0] * x[9]) * 0.0340 + (x[9] * x[2] * x[1]) * 0.0257 + cos(x[6]) * 0.092) / static_cast<double>(complexity);
            }
            break;
        case 8:
            for(int j = 0; j < complexity; j++){
                acc = acc + (sqrt(x[6] * x[6] + 0.01) * 0.267 + (x[4] * x[9] * x[3]) * 0.0145 + (x[3] + x[5]) * 0.257 + (x[3] * x[0]) * 0.0123 + (x[3] * x[0]) * 0.0053) / static_cast<double>(complexity);
            }
            break;
        case 9:
            for(int j = 0; j < complexity; j++){
                acc = acc + (sqrt(x[1] * x[1] + 0.01) * 0.203 + sin(x[0]) * 0.079 + cos(x[2]) * 0.246 + (x[5] * x[4] * x[9]) * 0.0102 + cos(x[4]) * 0.159) / static_cast<double>(complexity);
            }
            break;
        default:
            break;
        }
        return acc;
    };

    autodiff::forward::CudaFunctionWrapper<double> createcudafn() {
        autodiff::forward::CudaFunctionWrapper<double> cudafun;
        cudafun.register_fn_host<cu_f0>();
        return cudafun;
    }
    #endif
} // namespace testfun

#endif // EXAMPLE_FUNCTIONS_HPP_
