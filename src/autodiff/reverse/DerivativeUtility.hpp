#ifndef __DERIVATIVEUTILITY_HPP__
#define __DERIVATIVEUTILITY_HPP__

#include <Eigen/Core>
#include <functional>
#include "NodeManager.hpp"
#include "EigenSupport.hpp"
#include "Var.hpp"

/**
 * Utility functions for computing gradients and jacobians
 * for functions of 'Var<double>'s
 */

namespace autodiff {
namespace reverse {

void gradient(
    std::function<Var<double>(Eigen::Vector<Var<double>, Eigen::Dynamic>)> f,
    Eigen::Vector<double, Eigen::Dynamic> const & x,
    double & f_x,
    Eigen::Vector<double, Eigen::Dynamic> & grad
) {
    using VecVar = Eigen::Vector<Var<double>, Eigen::Dynamic>;
    using Var = Var<double>;
    using NodeManager = NodeManager<double>;

    VecVar var_x(x.size());
    
    for(size_t i = 0; i < var_x.size(); ++i) {
        var_x(i) = Var(x(i));
    }

    Var y = f(var_x);
    y.backward();

    f_x = y.value();

    grad.resizeLike(var_x);
    for(size_t i = 0; i < grad.size(); ++i) {
        grad(i) = var_x(i).grad();
    }

    NodeManager::instance().clear();
}

void jacobian(
    std::function<Eigen::Vector<Var<double>, Eigen::Dynamic>(Eigen::Vector<Var<double>, Eigen::Dynamic>)> f,
    Eigen::Vector<double, Eigen::Dynamic> const & x,
    Eigen::Vector<double, Eigen::Dynamic> & f_x,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> & jac
) {
    using VecVar = Eigen::Vector<Var<double>, Eigen::Dynamic>;
    using Var = Var<double>;
    using NodeManager = NodeManager<double>;

    VecVar var_x(x.size());

    for(size_t i = 0; i < var_x.size(); ++i) {
        var_x(i) = Var(x(i));
    }

    VecVar y = f(var_x);

    f_x.resizeLike(y);
    jac.resize(y.size(), var_x.size());
    for(size_t i = 0; i < y.size(); ++i) {
        f_x(i) = y(i).value();
        y(i).backward();

        // fill the i-th row of the jacobian
        for(size_t j = 0; j < var_x.size(); ++j) {
            jac(i,j) = var_x(j).grad();
        }
        
        // reset grad info for the next backward pass
        NodeManager::instance().clear_grad();
    }
    NodeManager::instance().clear();
}

}; // namespace reverse 
}; // namespace autodiff

#endif // __DERIVATIVEUTILITY_HPP__
