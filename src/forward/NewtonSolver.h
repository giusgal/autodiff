// NewtonSolver.h
#ifndef __NEWTON_SOLVER__H__
#define __NEWTON_SOLVER__H__

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cassert>
#include <numeric>
#include "DualVar.h"

namespace autodiff {
namespace forward {

DualVar<double> derivative(std::function<DualVar<double>(DualVar<double>)>f, double x0){
    DualVar<double> res = f(DualVar<double>(x0, 1.0));
    return res;
}

std::vector<double> gradient(std::function<DualVar<double>(std::vector<DualVar<double>>)>f,
    std::vector<double> x) {

        std::vector<DualVar<double>> xd;
        std::vector<double> res;

        xd.reserve(size(x));
        res.reserve(size(x));

        for(int i = 0; i < size(x); i++){
            xd.push_back(DualVar<double>(x[i], 0.0));
        }

        for(int i = 0; i < size(x); i++){
            xd[i].setInf(1.0);
            res.push_back(f(xd).getInf());
            xd[i].setInf(0.0);
        }

        return res;
}

Eigen::MatrixXd jacobian(std::function<std::vector<DualVar<double>>(std::vector<DualVar<double>>)>f, Eigen::VectorXd x0, Eigen::VectorXd &eval) {

    std::vector<DualVar<double>> inputs, res;
    int M = eval.size();
    int N = x0.size();
    inputs.reserve(N);

    // Initialize inputs with the values from x0 and seed value set to zero
    for (int i = 0; i < N; i ++) {
        inputs.emplace_back(DualVar<double>(x0[i], 0.0));
    }

    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(M, N);

    // Compute each column of the Jacobian
    // For each column of the jacobian, we are evaluating the function at point x0
    for (int i = 0; i < N; i++) {
        inputs[i].setInf(1.0);
        res = f(inputs);
        for (int j = 0; j < M; j++) {
            jacobian(j, i) = res[j].getInf();
        }
        inputs[i].setInf(0.0);
    }

    // recycle the last function evaluation
    for (int i = 0; i < M; i++) {
        eval[i] = res[i].getReal();
    }

    return jacobian;
}

Eigen::VectorXd solve(std::function<std::vector<DualVar<double>>(std::vector<DualVar<double>>)>f,
                        Eigen::VectorXd x0, int M, Eigen::VectorXd &f_eval) {

        // Matrix containing all partial derivatives of function f at point x = x0
        Eigen::MatrixXd J = jacobian(f, x0, f_eval);

        for (auto xi : f_eval) {
            if (std::isinf(xi) or std::isnan(xi)) throw std::overflow_error("function valuation is NaN");
        }

        for (auto xi : f_eval) {
            assert(not(std::isnan(xi)));
        }
        // Solve the linear system J * u = f_eval for u
        return J.fullPivLu().solve(f_eval);
}

Eigen::VectorXd newton(std::function<std::vector<DualVar<double>>(std::vector<DualVar<double>>)>f,
                        Eigen::VectorXd x0, int M, int maxit=1000, double tol=1e-6, bool v = false){

        Eigen::VectorXd x, x1, eval(M);
        x = x0;
        int i = 0;
        for(; i < maxit; i++){
            try{
                x1 = solve(f, x, M, eval);
            }
            catch (const std::overflow_error & e){
                std::cout << e.what() << std::endl;
                i = maxit;
                break;
            }
            x = x - x1;
            if (v) {
                std::cout << "iteration: " << i << std::endl;
                std::cout << "new guess:\n" << x << std::endl;
            }
            x1 = x1.cwiseAbs();
            eval = eval.cwiseAbs();
            double step_size = std::accumulate(x1.data(), x1.data() + x1.size(), 0.0);
            double residual = std::accumulate(eval.data(), eval.data() + eval.size(), 0.0);
            if (step_size < tol && residual < tol)
                break;
        }
        if (i == maxit) {
            throw std::runtime_error("Unable to converge");
        }
        if (v) {
            std::cout << "number of iterations: " << i << std::endl;
            std::cout << "x:\n" << x << std::endl;
        }
        for (auto xi : eval) {
            assert(not (std::isnan(xi) || std::isinf(xi)));
        }
        // double check
        std::vector<DualVar<double>> x_d(x.size()), x_d1(x.size());
        for(auto xi : x) {
            x_d.push_back(xi);
        }
        auto res_c = f(x_d);
        auto x2 = x + x1;

        for(auto xi : x2) {
            x_d1.push_back(xi);
        }
        auto res_c2 = f(x_d1);
        return x;
    }

}; // namespace forward
}; // namespace autodiff

#endif // __NEWTON_SOLVER__H__
