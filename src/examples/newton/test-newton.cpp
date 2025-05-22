#include <iostream>
#include <functional>
#include <Eigen/Core>
#include "Newton.hpp"
#include <chrono>

#define DIM 3
#define FN_OUTPUTS 2


using dv = autodiff::forward::DualVar<double>;
using dvec = Eigen::Matrix<dv, Eigen::Dynamic, 1>;

dvec fun_j(const dvec &x){
    dvec res(2);
    res << 5.0 * x(0) * x(0) + x(1) * x(1) * x(0) + sin(2.0 * x(1)) * sin(2.0 * x(1)) - 2.0,
         autodiff::forward::pow(std::exp(1), 2.0 * x(0)-x(1)) + 4.0 * x(1) - 3.0;

    return res;
}

std::function<dvec(const dvec &)> create_test_fn(int N, int M) {

    std::vector<int> ops(N);

    for (int i = 0; i < N; i++) {
        ops[i] = rand() % 5;
    }
    std::vector<int> rand_idxs(N);

    for(int i = 0; i < rand_idxs.size(); i++) {
        rand_idxs[i] = rand() % N;
    }

    return [=] (const dvec & x) -> dvec {
        dvec fx(N);
        auto getrandidx = [i = 0, &rand_idxs, N]() mutable { i++; return rand_idxs[i % N];};
        for (int i = 0; i < N; i++) {
            switch (ops[i] % 5) {
                case 0: fx[i] = x[getrandidx()] * x[getrandidx()] + pow(x[getrandidx()], 2.0); break;
                case 1: fx[i] = sin(x[getrandidx()] * cos(x[getrandidx()])); break;
                case 2: fx[i] = cos(x[getrandidx()]) - 1.0; break;
                case 3: {
                  auto x1 = x[getrandidx()];
                  auto x2 = x[getrandidx()];
                  auto x3 = cos(x[getrandidx()]);
                  fx[i] = x1 / (x2 * x3); break;
                }
                // fx[i] = x[getrandidx()] / (x[getrandidx()] * cos(x[getrandidx()])); break;
                case 4: fx[i] = pow(x[getrandidx()], 3.0) * x[getrandidx()];
            }
        }
        return fx;
    };
}

void test_newton(int n_runs, newton::NewtonOpts opts, std::function<dvec(const dvec &)> * fun_array){
  for (int i = 0; i < n_runs; i++) {
    auto fn = fun_array[i];
    newton::Newton<double> nsolver(fn, opts);
    Eigen::VectorXd init_guess = Eigen::VectorXd::Random(opts.dim_in);
    auto res = nsolver.solve(init_guess);
    
    // test that f(res) is approx. 0
    dvec res_dual(opts.dim_in), eval(opts.dim_out);
    for(int i = 0; i < opts.dim_in; i++) {
      res_dual[i] = dv(res[i]);
    }
    eval = fn(res_dual);

    for(int i = 0; i < opts.dim_out; i++) {
      // assert(eval[i].getReal() < 0.001);
      ;
    }

  }
}


int main() {

  using std::chrono::high_resolution_clock;
  using std::chrono::duration;
  using std::chrono::milliseconds;
  
  int nruns = 2;
  int dim_in = 200;
  int dim_out = 200;
  std::function<dvec(const dvec &)> fun_array[nruns];

  for(int i = 0; i < nruns; i++) {
    fun_array[i] = create_test_fn(dim_in, dim_out);
  }

  auto nonLinFun = [](const dvec &x) -> dvec {
    dvec res(2);
    res << 5.0 * x(0) * x(0) + x(1) * x(1) * x(0) + sin(2.0 * x(1)) * sin(2.0 * x(1)) - 2.0,
         autodiff::forward::pow(std::exp(1), 2.0 * x(0)-x(1)) + 4.0 * x(1) - 3.0;

    return res;
  };

  newton::NewtonOpts newtonopts;
  newtonopts.maxit = 100;
  newtonopts.tol = 1e-6;
  newtonopts.dim_in = dim_in;
  newtonopts.dim_out = dim_out;
  newtonopts.parallel = 0;


  auto tstart = high_resolution_clock::now();
  test_newton(nruns, newtonopts, fun_array);
  auto tend = high_resolution_clock::now();
  duration<double, std::milli> elapsed = tend - tstart;
  std::cout << "Time taken with sequential Jacobian " << elapsed.count() << std::endl;

  newtonopts.parallel = 1;
  tstart = high_resolution_clock::now();
  test_newton(nruns, newtonopts, fun_array);
  tend = high_resolution_clock::now();
  elapsed = tend - tstart;
  std::cout << "Time taken with parallel Jacobian " << elapsed.count() << std::endl;

}





