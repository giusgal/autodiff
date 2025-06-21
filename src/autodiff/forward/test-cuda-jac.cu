// test_jacobian_cuda.cpp
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include "DualVar.hpp"
#include "ForwardDifferentiator.hpp"
#include "CudaSupport.hpp"
#include "example-functions.hpp"

using dv = autodiff::forward::DualVar<double>;
using dvec = Eigen::Matrix<dv, Eigen::Dynamic, 1>;

int main() {
    using Clock = std::chrono::high_resolution_clock;

    // Problem dimensions
    int dim_in = testfun::input_dim;   // Input dimension
    int dim_out = testfun::output_dim;  // Output dimension

    // Create input vector
    Eigen::VectorXd x0 = Eigen::VectorXd::Random(dim_in);
    Eigen::VectorXd real_eval_cpu(dim_out);

    // Create Jacobian object
    Eigen::MatrixXd j(dim_out, dim_in);
    

    // Test regular CPU compute
    auto t1 = Clock::now();
    autodiff::forward::jacobian<double>(testfun::test_fun, x0, real_eval, j);
    auto t2 = Clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "CPU Jacobian norm:\n" << jacobian.getJacobian().norm() << std::endl;
    std::cout << "CPU Time: " << cpu_time << " μs\n\n";

#ifdef USE_CUDA
    Eigen::MatrixXd jc(dim_out, dim_in);
    Eigen::VectorXd real_eval(dim_out);

    autodiff::forward::CudaFunctionWrapper<double> cudafun = testfun::createcudafn();
    // Test CUDA compute
    auto t3 = Clock::now();
    autodiff::forward::jacobian_cuda<double>(cudafun, x0, real_eval, jc);
    auto t4 = Clock::now();
    auto cuda_time = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    std::cout << "CUDA Jacobian norm:\n" << jc.norm() << std::endl;
    std::cout << "CUDA Time: " << cuda_time << " μs\n\n";

    // Calculate speedup
    double speedup = static_cast<double>(cpu_time) / cuda_time;
    std::cout << "CUDA Speedup: " << speedup << "x\n";
#endif

    return 0;
}