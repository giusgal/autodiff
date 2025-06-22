#include <Eigen/Core>
#include <gtest/gtest.h>
#include "Var.hpp"
#include "ReverseUtility.hpp"

/**
 * Unit tests for the utility functions exposed by
 *  ReverseUtility.hpp
 */

using Var = autodiff::reverse::Var<double>;
using VecVar = Eigen::Vector<Var, Eigen::Dynamic>;
using Vec = Eigen::Vector<double, Eigen::Dynamic>;
using Jac = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

void gradient_finite_diff(
    std::function<double(Vec const &)> f,
    Vec const & x,
    double & f_x,
    Vec & grad
) {
    constexpr double incr = 0.000001;

    grad.resizeLike(x);
    for(size_t i = 0; i < x.size(); ++i) {
        Vec h = Vec::Zero(x.size());
        h(i) = incr;

        grad(i) = (f(x + h) - f(x))/incr;
    }
}

void jacobian_finite_diff(
    std::function<Vec(Vec const &)> f,
    Vec const & x,
    Vec & f_x,
    Jac & jac
) {
    constexpr double incr = 0.000001;

    f_x = f(x);

    jac.resize(f_x.size(), x.size());
    for(size_t j = 0; j < x.size(); ++j) {
        Vec h = Vec::Zero(x.size());
        h(j) = incr;

        jac.block(0,j,jac.rows(),1) = (f(x + h) - f(x))/incr;
    }
}

template <typename OutType, typename InType>
OutType f_N1_1(InType const & x) {
    // 2 inputs
    return sin(x(0)) + log(1.0 + exp(x(1))) + 0.5 * x(0) * x(1);
}

template <typename OutType, typename InType>
OutType f_NM_1(InType const & x) {
    OutType res(2);

    res <<
        sin(x(0)) + log(1.0 + exp(x(1))) + 0.5 * x(0) * x(1),
        tanh(x(1)) + sqrt(abs(x(1))) * x(0) * x(1);

    return res;
}

TEST(ReverseUtilityTest, GradientTest1) {
    Vec grad_fd;
    Vec grad_ad;
    Vec x = Vec::Ones(2);
    double f_x;

    gradient_finite_diff(f_N1_1<double, Vec>, x, f_x, grad_fd);
    autodiff::reverse::gradient(f_N1_1<Var, VecVar>, x, f_x, grad_ad);

    double eps = 0.0001;
    for(size_t i = 0; i < grad_ad.size(); ++i) {
        ASSERT_NEAR(grad_fd(i), grad_ad(i), eps);
    }
}

TEST(ReverseUtilityTest, JacobianTest1) {
    Jac jac_fd;
    Jac jac_ad;
    Vec x = Vec::Ones(2);
    Vec f_x;

    jacobian_finite_diff(f_NM_1<Vec, Vec>, x, f_x, jac_fd);
    autodiff::reverse::jacobian(f_NM_1<VecVar, VecVar>, x, f_x, jac_ad);

    double eps = 0.0001;
    for(size_t i = 0; i < jac_ad.rows(); ++i) {
        for(size_t j = 0; j < jac_ad.cols(); ++j) {
            ASSERT_NEAR(jac_fd(i,j), jac_ad(i,j), eps);
        }
    }
}
