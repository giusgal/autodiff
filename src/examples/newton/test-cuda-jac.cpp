// test_jacobian_cuda.cpp
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include "../../autodiff/forward/CudaSupport.hpp"
#include "example-functions.hpp"
#include "../../autodiff/forward/autodiff.hpp"

using dv = autodiff::forward::DualVar<double>;
using dvec = Eigen::Matrix<dv, Eigen::Dynamic, 1>;

int main() {
    using Clock = std::chrono::high_resolution_clock;

    // Problem dimensions
    int dim_in = 8;   // Input dimension
    int dim_out = 256;  // Output dimension

    // Create input point
    Eigen::VectorXd x0 = Eigen::VectorXd::Random(dim_in);
    Eigen::VectorXd real_eval_cpu(dim_out);

    // Create Jacobian object
    newton::ForwardJac<double> jacobian(dim_out, dim_in, testfun::test_fun);

    // Test regular CPU compute
    auto t1 = Clock::now();
    jacobian.compute(x0, real_eval_cpu);
    auto t2 = Clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    std::cout << "CPU Jacobian:\n" << jacobian.getJacobian() << std::endl;
    std::cout << "CPU Function Value: " << real_eval_cpu.transpose() << std::endl;
    std::cout << "CPU Time: " << cpu_time << " μs\n\n";

#ifdef USE_CUDA

    newton::CudaFunctionWrapper<double> cudafun = testfun::createwrapper();

    newton::CudaJac<double> jacobian_cu(dim_out, dim_in, cudafun);
    // Test CUDA compute
    auto t3 = Clock::now();
    jacobian_cu.compute(x0);
    auto t4 = Clock::now();
    auto cuda_time = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();

    std::cout << "CUDA Jacobian:\n" << jacobian_cu.getJacobian() << std::endl;
    std::cout << "CUDA Time: " << cuda_time << " μs\n\n";

    // Calculate speedup
    double speedup = static_cast<double>(cpu_time) / cuda_time;
    std::cout << "CUDA Speedup: " << speedup << "x\n";
#endif

    return 0;
}