#include <iostream>
#include <functional>
#include <Eigen/Core>
#include "autodiff.hpp"
#include "EigenSupport.hpp"
#include "../../examples/newton/Newton.hpp"

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

// std::function<dvec(const dvec &)> create_test_fn(int M, int N) {

//     std::vector<int> ops(M);

//     for (int i = 0; i < M; i++) {
//         ops[i] = rand() % 5;
//     }
//     std::vector<int> rand_idxs(20);

//     for(int i = 0; i < rand_idxs.size(); i++) {
//         rand_idxs[i] = rand() % N;
//     }

//     return [=] (const dvec & x) -> dvec {
//         dvec fx;
//         auto getrandidx = [i = 0, &rand_idxs]() mutable { i++; return rand_idxs[i];};
//         for (int i = 0; i < M; i++) {
//             switch (ops[i] % 5) {
//                 case 0: fx.row(i) << x[getrandidx()] * x[getrandidx()] + pow(x[getrandidx()], 2.0); break;
//                 case 1: fx.row(i) << sin(x[getrandidx()] * cos(x[getrandidx()])); break;
//                 case 2: fx.row(i) << cos(x[getrandidx()]) - 1.0; break;
//                 case 3: fx.row(i) << x[getrandidx()] / (x[getrandidx()] * cos(x[getrandidx()])); break;
//                 case 4: fx.row(i) << pow(x[getrandidx()], 3.0) * x[getrandidx()];
//             }
//         }
//         return fx;
//     };
// }


int main() {
    auto nonLinFun = [](const dvec &x) -> dvec {
    dvec res(2);
    res << 5.0 * x(0) * x(0) + x(1) * x(1) * x(0) + sin(2.0 * x(1)) * sin(2.0 * x(1)) - 2.0,
         autodiff::forward::pow(std::exp(1), 2.0 * x(0)-x(1)) + 4.0 * x(1) - 3.0;

    return res;
  };
    newton::NewtonOpts newtonopts;
    newtonopts.maxit = 100;
    newtonopts.tol = 1e-6;
    newtonopts.dim_in = 2;
    newtonopts.dim_out = 2;
    newton::Newton<double> nsolver(nonLinFun, newtonopts);
    Eigen::VectorXd v(2);
    v << 1.0, 1.0;

    auto res = nsolver.solve(v);
    std::cout << "Newton result: " << res.transpose() << std::endl;
     
}





