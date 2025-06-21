#include <iostream>
#include <Eigen/Core>
#include <chrono>
#include "DualVar.hpp"
#include "ForwardDifferentiator.hpp"

using dv = autodiff::forward::DualVar<double>;
using dvec = Eigen::Matrix<dv, Eigen::Dynamic, 1>;
using RealVec = Eigen::VectorXd;

// Example function: f(x) = [x0^2 + x1, sin(x0 * x1)]
dvec test_fun(const dvec &x) {
  dvec res(2);
  res[0] = x[0] * x[0] + x[1];
  res[1] = sin(x[0] * x[1]);
  return res;
}

std::function<dvec(const dvec&)> make_big_test_fun(int N, int M) {
    return [N, M](const dvec& x) {
        dvec res(M);
        for (int i = 0; i < M; ++i) {
            res[i] = dv(0.0, 0.0);
            for (int j = 0; j < N; ++j) {
                res[i] = res[i] + sin(x[j] + i) * cos(x[j] * (i+1));
            }
            // nonlinear
            res[i] = res[i] + exp(x[i % N]);
        }
        return res;
    };
}


int main() {
  using Clock = std::chrono::high_resolution_clock;
  using ms = std::chrono::milliseconds;

  int dim_in = 1000;
  int dim_out = 1000;
  RealVec x0 = RealVec::Random(dim_in);
  RealVec real_eval(dim_out);

  auto big_fun = make_big_test_fun(dim_in, dim_out);

  // Sequential
  auto t1 = Clock::now();
  Eigen::MatrixXd j(dim_out, dim_in);
  autodiff::forward::jacobian<double>(big_fun, x0, real_eval, j);
  auto t2 = Clock::now();
  std::cout << "Sequential Jacobian norm:\n" << j.norm() << std::endl;
  std::cout << "Function value norm: " << real_eval.transpose().norm() << std::endl;
  std::cout << "Sequential time: " << std::chrono::duration_cast<ms>(t2 - t1).count() << " ms\n";

  return 0;
}