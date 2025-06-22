#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include "ForwardUtility.hpp"
#include "DualVar.hpp"

using namespace autodiff::forward;

class fwdiff : public ::testing::Test {
protected:
    void SetUp() override {
        // Test values
        x = 3.0; 
        eps = 1e-10;
    }

    double x, y, z;
    double eps;
};

template <typename T>
DualVar<T> poly_test(DualVar<T> x) {
  return 12.0 * x * x + 3.0 * x + 4;
}

// polynomial: f(x,y) = x^2 + 2*x*y + y^2 + 3*x + 4*y + 5
template <typename T>
DualVar<T> multi_poly_test(std::vector<DualVar<T>> vars) {
    auto x = vars[0];
    auto y = vars[1];
    return x * x + 2.0 * x * y + y * y + 3.0 * x + 4.0 * y + 5.0;
}

// polynomial using Eigen vectors
template <typename T>
DualVar<T> multi_poly_eigen_test(DualVec<T> vars) {
    auto x = vars[0];
    auto y = vars[1];
    return x * x + 2.0 * x * y + y * y + 3.0 * x + 4.0 * y + 5.0;
}

// [x^2 + y, x + y^2]
template <typename T>
DualVec<T> vector_function_test(DualVec<T> vars) {
    DualVec<T> result(2);
    auto x = vars[0];
    auto y = vars[1];
    result[0] = x * x + y;
    result[1] = x + y * y;
    return result;
}

TEST_F(fwdiff, simple) {
    std::function<DualVar<double>(DualVar<double>)> f = poly_test<double>;
    double res = derivative(f, x);
    EXPECT_EQ(res, 75.0);
}

TEST_F(fwdiff, gradient_std_vector) {
    // Test gradient using std::vector
    std::function<DualVar<double>(std::vector<DualVar<double>>)> f = multi_poly_test<double>;
    std::vector<double> point = {2.0, 3.0};
    
    std::vector<double> grad = gradient(f, point);
    
    EXPECT_EQ(grad.size(), 2);
    EXPECT_NEAR(grad[0], 13.0, eps);
    EXPECT_NEAR(grad[1], 14.0, eps);
}

TEST_F(fwdiff, gradient_eigen_vector) {
    // Test gradient using Eigen vectors
    std::function<DualVar<double>(DualVec<double>)> f = multi_poly_eigen_test<double>;
    RealVec<double> point(2);
    point << 2.0, 3.0;
    
    RealVec<double> grad = gradient(f, point);
    
    EXPECT_EQ(grad.size(), 2);
    EXPECT_NEAR(grad[0], 13.0, eps);
    EXPECT_NEAR(grad[1], 14.0, eps);
}

TEST_F(fwdiff, jacobian_calculation) {
    // Test Jacobian computation
    std::function<DualVec<double>(DualVec<double>)> f = vector_function_test<double>;
    
    RealVec<double> point(2);
    point << 2.0, 3.0;
    
    RealVec<double> f_x(2);  // Function values at point
    JacType<double> jac(2, 2);  // Jacobian matrix
    
    jacobian(f, point, f_x, jac);
    
    // Expected function values at (2,3): [2^2 + 3, 2 + 3^2] = [7, 11]
    EXPECT_NEAR(f_x[0], 7.0, eps);
    EXPECT_NEAR(f_x[1], 11.0, eps);
    

    EXPECT_NEAR(jac(0, 0), 4.0, eps);  
    EXPECT_NEAR(jac(0, 1), 1.0, eps);  
    EXPECT_NEAR(jac(1, 0), 1.0, eps);  
    EXPECT_NEAR(jac(1, 1), 6.0, eps);  
}

TEST_F(fwdiff, gradient_single_variable) {
    // Test gradient for single variable (should behave like derivative)
    std::function<DualVar<double>(std::vector<DualVar<double>>)> f = [](std::vector<DualVar<double>> vars) {
        return poly_test(vars[0]);
    };
    
    std::vector<double> point = {3.0};
    std::vector<double> grad = gradient(f, point);
    
    EXPECT_EQ(grad.size(), 1);
    EXPECT_NEAR(grad[0], 75.0, eps);
}

TEST_F(fwdiff, jacobian_single_output) {
    // Test Jacobian for scalar-valued function (should behave like gradient)
    std::function<DualVec<double>(DualVec<double>)> f = [](DualVec<double> vars) {
        DualVec<double> result(1);
        result[0] = multi_poly_eigen_test(vars);
        return result;
    };
    
    RealVec<double> point(2);
    point << 2.0, 3.0;
    
    RealVec<double> f_x(1);
    JacType<double> jac(1, 2);
    
    jacobian(f, point, f_x, jac);
    
    // Expected function value at (2,3): 2^2 + 2*2*3 + 3^2 + 3*2 + 4*3 + 5 = 4 + 12 + 9 + 6 + 12 + 5 = 48
    EXPECT_NEAR(f_x[0], 48.0, eps);
    
    // Expected gradient: [13, 14]
    EXPECT_NEAR(jac(0, 0), 13.0, eps);
    EXPECT_NEAR(jac(0, 1), 14.0, eps);
}