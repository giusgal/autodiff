#include <iostream>
#include <functional>
#include <Eigen/Core>
#include "autodiff.hpp"

using dv = autodiff::forward::DualVar<double>;
using dvec = std::vector<dv>;


dv fun(dv x){
    return 3.0 * log(x) + 2.0;
}

dv fun_v(dvec x){
    return 16.0 * x[0] + 32.0 * x[1] + x[2] * x[3];
}

dvec fun_j(dvec x){
    dvec res;
    res.reserve(2);

    res.push_back(5.0 * x[0] * x[0] + x[1] * x[1] * x[0] + sin(2.0 * x[1]) * sin(2.0 * x[1]) - 2.0);
    res.push_back(autodiff::forward::pow(std::exp(1), 2.0 * x[0]-x[1]) + 4.0 * x[1] - 3.0);

    return res;
}

std::function<dvec(const dvec &)> create_test_fn(int M, int N) {

    std::vector<int> ops(M);

    for (int i = 0; i < M; i++) {
        ops[i] = rand() % 5;
    }
    std::vector<int> rand_idxs(20);

    for(int i = 0; i < rand_idxs.size(); i++) {
        rand_idxs[i] = rand() % N;
    }

    // return [=] (const dvec & x) -> dvec {
    //     dvec fx;
    //     for (int i = 0; i < dim; i++) {
    //         dv xi = x[i] - root[i];
    //         switch (ops[i] % 5) {
    //             case 0: fx.push_back(xi * xi); break;
    //             case 1: fx.push_back(sin(xi)); break;
    //             case 2: fx.push_back(cos(xi) - 1.0); break;
    //             case 3: fx.push_back(); break;
    //             case 4: fx.push_back(xi * xi * xi);
    //         }
    //     }
    //     return fx;
    // };


    return [=] (const dvec & x) -> dvec {
        dvec fx;
        auto getrandidx = [i = 0, &rand_idxs]() mutable { i++; return rand_idxs[i];};
        for (int i = 0; i < M; i++) {
            switch (ops[i] % 5) {
                case 0: fx.push_back(x[getrandidx()] * x[getrandidx()] + pow(x[getrandidx()], 2.0)); break;
                case 1: fx.push_back(sin(x[getrandidx()] * cos(x[getrandidx()]))); break;
                case 2: fx.push_back(cos(x[getrandidx()]) - 1.0); break;
                case 3: fx.push_back(x[getrandidx()] / (x[getrandidx()] * cos(x[getrandidx()]))); break;
                case 4: fx.push_back(pow(x[getrandidx()], 3.0) * x[getrandidx()]);
            }
        }
        return fx;
    };
}

dvec create_random_dvec(int dim) {
    dvec root;
    for(int i = 0; i < dim; i++) {
        dv xi(static_cast <float> (rand() / static_cast <float> (RAND_MAX)) * 10.0 - 5.0);
        root.push_back(xi);
    }
    return root;
}

void test_newton(int n, int maxdim, int nvars) {
    Eigen::VectorXd x0, res;
    int dim;
    int succ = 0;
    for (int i = 0; i < n; i++) { 
        dim = maxdim;
        auto fn = create_test_fn(dim, nvars);

        
        x0 = Eigen::VectorXd::Random(dim) * 5.0;
        try {
            res = newton(fn, x0, dim, 1000, 1.0e-6, true);
        } catch (const std::runtime_error & e ) {
            std::cout << e.what() << std::endl;
            continue;
        }
        dvec res_dual;
        for (auto xi : res) {
            res_dual.push_back(dv(xi));
        }

        dvec eval = fn(res_dual);
        int discard = 0;

        for (auto xi : eval) {
            assert(abs(xi).getReal() < 0.00001);
        }
        if (discard) continue;
        succ ++;

        
    }
    std::cout << "converged " << succ << " times, or " << 
            static_cast<float> (static_cast<float> (succ) / static_cast<float> (n)) << " of total runs" << std::endl;
        
}


using namespace std;
int main() {

    Eigen::VectorXd v(2);
    v << 1.0, 1.0;

    //Eigen::VectorXd res = newton(fun_j, v, 2);
    dv r0, r1;
    
    test_newton(1000, 5, 5);   
}




