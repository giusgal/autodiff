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

template <typename OutType, typename InType>
OutType f_N1_1(InType const & x) {
    // 2 inputs
    return sin(x(0)) + log(1.0 + exp(x(1))) + 0.5 * x(0) * x(1);
}

// template <typename OutType, typename InType>
// OutType f_N1_2(InType const & x) {
//     // 42 inputs
//     OutType y = 0.0;
//
//     y += sin(x(0)) * cos(x(1)) + tan(x(2)) / (1.0 + exp(-x(3)));
//     y += log(1.0 + exp(x(4))) * sqrt(abs(x(5))) - 0.5 * x(6) * x(7);
//     y += pow(x(8), 2.0) + pow(x(9), 3.0) - exp(-x(10)) * sin(x(11));
//     y += sqrt(pow(x(12), 2.0) + pow(x(13), 2.0)) * cos(x(14));
//     y += log(abs(x(15)) + 1.0) * tanh(x(16)) + pow(x(17), x(18));
//     y += sin(x(19) * x(20)) + cos(x(21) + x(22)) - x(23) / (1.0 + abs(x(24)));
//     y += exp(x(25)) * log(x(26) * x(27) + 1.0) - pow(x(28), 0.5) * tan(x(29));
//     y += 0.25 * pow(sin(x(30) + x(31)), 2.0) + 0.75 * cos(x(32) * x(33));
//     y += tanh(x(34)) * sqrt(abs(x(35))) + pow(x(36) + x(37), 1.5);
//     y += log(1.0 + exp(sin(x(38) * cos(x(39))))) * (x(40) - x(41));
//
//     return y;
// }


template <typename OutType, typename InType>
OutType f_NM(InType const & x) {
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

// TEST(ReverseUtilityTest, GradientTest2) {
//     Vec grad_fd;
//     Vec grad_ad;
//     Vec x = Vec::Ones(42);
//     double f_x;
//
//     gradient_finite_diff(f_N1_2<double, Vec>, x, f_x, grad_fd);
//     autodiff::reverse::gradient(f_N1_1<Var, VecVar>, x, f_x, grad_ad);
//
//     double eps = 0.0001;
//     for(size_t i = 0; i < grad_ad.size(); ++i) {
//         ASSERT_NEAR(grad_fd(i), grad_ad(i), eps) << "Test failed at i = " << i;
//     }
// }
