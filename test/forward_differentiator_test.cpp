#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include "ForwardDifferentiator.hpp"
#include "DualVar.hpp"

using namespace autodiff::forward;

class ForwardDifferentiatorTest : public ::testing::Test {
protected:
  void SetUp() override {
    eps = 1e-10;
  }

  double eps;
};

// Test derivative function for single variable
TEST_F(ForwardDifferentiatorTest, DerivativeSingleVariable) {
  // Test f(x) = x^2, f'(x) = 2x
  auto f1 = [](DualVar<double> x) { return x * x; };
  auto result1 = derivative(f1, 3.0);
  EXPECT_NEAR(result1.getReal(), 9.0, eps);  // f(3) = 9
  EXPECT_NEAR(result1.getInf(), 6.0, eps);   // f'(3) = 6

  // Test f(x) = sin(x), f'(x) = cos(x)
  auto f2 = [](DualVar<double> x) { return sin(x); };
  auto result2 = derivative(f2, M_PI/4);
  EXPECT_NEAR(result2.getReal(), std::sin(M_PI/4), eps);
  EXPECT_NEAR(result2.getInf(), std::cos(M_PI/4), eps);

  // Test f(x) = exp(x), f'(x) = exp(x)
  auto f3 = [](DualVar<double> x) { return exp(x); };
  auto result3 = derivative(f3, 1.0);
  EXPECT_NEAR(result3.getReal(), std::exp(1.0), eps);
  EXPECT_NEAR(result3.getInf(), std::exp(1.0), eps);

  // Test f(x) = log(x), f'(x) = 1/x
  auto f4 = [](DualVar<double> x) { return log(x); };
  auto result4 = derivative(f4, 2.0);
  EXPECT_NEAR(result4.getReal(), std::log(2.0), eps);
  EXPECT_NEAR(result4.getInf(), 0.5, eps);
}

// Test gradient function with std::vector
TEST_F(ForwardDifferentiatorTest, GradientStdVector) {
  // Test f(x,y) = x^2 + y^2, gradient = (2x, 2y)
  auto f1 = [](std::vector<DualVar<double>> vars) {
    return vars[0] * vars[0] + vars[1] * vars[1];
  };
  std::vector<double> x1 = {2.0, 3.0};
  auto grad1 = gradient(f1, x1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR(grad1[0], 4.0, eps);  // 2*2
  EXPECT_NEAR(grad1[1], 6.0, eps);  // 2*3

  // Test f(x,y,z) = x*y + y*z + x*z, gradient = (y+z, x+z, y+x)
  auto f2 = [](std::vector<DualVar<double>> vars) {
    return vars[0] * vars[1] + vars[1] * vars[2] + vars[0] * vars[2];
  };
  std::vector<double> x2 = {1.0, 2.0, 3.0};
  auto grad2 = gradient(f2, x2);
  EXPECT_EQ(grad2.size(), 3);
  EXPECT_NEAR(grad2[0], 5.0, eps);  // y+z = 2+3
  EXPECT_NEAR(grad2[1], 4.0, eps);  // x+z = 1+3
  EXPECT_NEAR(grad2[2], 3.0, eps);  // y+x = 2+1

  // Test constant function, gradient should be zero
  auto f3 = [](std::vector<DualVar<double>> vars) {
    return DualVar<double>(5.0);
  };
  std::vector<double> x3 = {1.0, 2.0};
  auto grad3 = gradient(f3, x3);
  EXPECT_EQ(grad3.size(), 2);
  EXPECT_NEAR(grad3[0], 0.0, eps);
  EXPECT_NEAR(grad3[1], 0.0, eps);
}

// Test gradient function with Eigen vectors
TEST_F(ForwardDifferentiatorTest, GradientEigenVector) {
  // Test f(x,y) = x^2 + y^2
  auto f1 = [](DualVec<double> vars) {
    return vars[0] * vars[0] + vars[1] * vars[1];
  };
  RealVec<double> x1(2);
  x1 << 2.0, 3.0;
  auto grad1 = gradient(f1, x1);
  EXPECT_EQ(grad1.size(), 2);
  EXPECT_NEAR(grad1[0], 4.0, eps);
  EXPECT_NEAR(grad1[1], 6.0, eps);

  // Test f(x,y,z) = sin(x) + cos(y) + exp(z)
  auto f2 = [](DualVec<double> vars) {
    return sin(vars[0]) + cos(vars[1]) + exp(vars[2]);
  };
  RealVec<double> x2(3);
  x2 << M_PI/2, 0.0, 0.0;
  auto grad2 = gradient(f2, x2);
  EXPECT_EQ(grad2.size(), 3);
  EXPECT_NEAR(grad2[0], std::cos(M_PI/2), eps);  // cos(pi/2) = 0
  EXPECT_NEAR(grad2[1], -std::sin(0.0), eps);    // -sin(0) = 0
  EXPECT_NEAR(grad2[2], std::exp(0.0), eps);     // exp(0) = 1
}

// Test jacobian function
TEST_F(ForwardDifferentiatorTest, JacobianFunction) {
  // Test vector function f(x,y) = [x^2 + y, x + y^2]
  auto f1 = [](DualVec<double> vars) {
    DualVec<double> result(2);
    result[0] = vars[0] * vars[0] + vars[1];
    result[1] = vars[0] + vars[1] * vars[1];
    return result;
  };
  
  RealVec<double> x(2);
  x << 2.0, 3.0;
  
  RealVec<double> f_x(2);
  JacType<double> jac(2, 2);
  
  jacobian(f1, x, f_x, jac);
  
  // Check function values
  EXPECT_NEAR(f_x[0], 7.0, eps);  // 2^2 + 3 = 7
  EXPECT_NEAR(f_x[1], 11.0, eps); // 2 + 3^2 = 11
  
  // Check Jacobian
  // df1/dx = 2x = 4, df1/dy = 1
  // df2/dx = 1, df2/dy = 2y = 6
  EXPECT_NEAR(jac(0, 0), 4.0, eps);
  EXPECT_NEAR(jac(0, 1), 1.0, eps);
  EXPECT_NEAR(jac(1, 0), 1.0, eps);
  EXPECT_NEAR(jac(1, 1), 6.0, eps);
}

// Test complex jacobian with trigonometric functions
TEST_F(ForwardDifferentiatorTest, JacobianTrigonometric) {
  // Test f(x,y) = [sin(x)*cos(y), cos(x)*sin(y)]
  auto f = [](DualVec<double> vars) {
    DualVec<double> result(2);
    result[0] = sin(vars[0]) * cos(vars[1]);
    result[1] = cos(vars[0]) * sin(vars[1]);
    return result;
  };
  
  RealVec<double> x(2);
  x << M_PI/4, M_PI/6;
  
  RealVec<double> f_x(2);
  JacType<double> jac(2, 2);
  
  jacobian(f, x, f_x, jac);
  
  double sin_pi4 = std::sin(M_PI/4);
  double cos_pi4 = std::cos(M_PI/4);
  double sin_pi6 = std::sin(M_PI/6);
  double cos_pi6 = std::cos(M_PI/6);
  
  // Check function values
  EXPECT_NEAR(f_x[0], sin_pi4 * cos_pi6, eps);
  EXPECT_NEAR(f_x[1], cos_pi4 * sin_pi6, eps);
  
  // Check Jacobian
  // d/dx[sin(x)*cos(y)] = cos(x)*cos(y)
  // d/dy[sin(x)*cos(y)] = sin(x)*(-sin(y))
  // d/dx[cos(x)*sin(y)] = -sin(x)*sin(y)
  // d/dy[cos(x)*sin(y)] = cos(x)*cos(y)
  EXPECT_NEAR(jac(0, 0), cos_pi4 * cos_pi6, eps);
  EXPECT_NEAR(jac(0, 1), -sin_pi4 * sin_pi6, eps);
  EXPECT_NEAR(jac(1, 0), -sin_pi4 * sin_pi6, eps);
  EXPECT_NEAR(jac(1, 1), cos_pi4 * cos_pi6, eps);
}

// Test jacobian with different dimensions
TEST_F(ForwardDifferentiatorTest, JacobianDifferentDimensions) {
  // Test 3D input to 2D output: f(x,y,z) = [x+y+z, x*y*z]
  auto f = [](DualVec<double> vars) {
    DualVec<double> result(2);
    result[0] = vars[0] + vars[1] + vars[2];
    result[1] = vars[0] * vars[1] * vars[2];
    return result;
  };
  
  RealVec<double> x(3);
  x << 1.0, 2.0, 3.0;
  
  RealVec<double> f_x(2);
  JacType<double> jac(2, 3);
  
  jacobian(f, x, f_x, jac);
  
  // Check function values
  EXPECT_NEAR(f_x[0], 6.0, eps);  // 1+2+3
  EXPECT_NEAR(f_x[1], 6.0, eps);  // 1*2*3
  
  // Check Jacobian
  // First row: [1, 1, 1]
  // Second row: [y*z, x*z, x*y] = [6, 3, 2]
  EXPECT_NEAR(jac(0, 0), 1.0, eps);
  EXPECT_NEAR(jac(0, 1), 1.0, eps);
  EXPECT_NEAR(jac(0, 2), 1.0, eps);
  EXPECT_NEAR(jac(1, 0), 6.0, eps);  // y*z = 2*3
  EXPECT_NEAR(jac(1, 1), 3.0, eps);  // x*z = 1*3
  EXPECT_NEAR(jac(1, 2), 2.0, eps);  // x*y = 1*2
}

// Test edge cases
TEST_F(ForwardDifferentiatorTest, EdgeCases) {
  // Test single variable, single output
  auto f1 = [](DualVec<double> vars) {
    DualVec<double> result(1);
    result[0] = vars[0] * vars[0];
    return result;
  };
  
  RealVec<double> x1(1);
  x1 << 5.0;
  
  RealVec<double> f_x1(1);
  JacType<double> jac1(1, 1);
  
  jacobian(f1, x1, f_x1, jac1);
  
  EXPECT_NEAR(f_x1[0], 25.0, eps);
  EXPECT_NEAR(jac1(0, 0), 10.0, eps);  // 2*5

  // Test linear function (jacobian should be constant)
  auto f2 = [](DualVec<double> vars) {
    DualVec<double> result(2);
    result[0] = 2.0 * vars[0] + 3.0 * vars[1];
    result[1] = vars[0] - 4.0 * vars[1];
    return result;
  };
  
  RealVec<double> x2(2);
  x2 << 1.0, 1.0;
  
  RealVec<double> f_x2(2);
  JacType<double> jac2(2, 2);
  
  jacobian(f2, x2, f_x2, jac2);
  
  // Check function values
  EXPECT_NEAR(f_x2[0], 5.0, eps);   // 2*1 + 3*1
  EXPECT_NEAR(f_x2[1], -3.0, eps);  // 1 - 4*1
  
  // Check Jacobian (should be constant)
  EXPECT_NEAR(jac2(0, 0), 2.0, eps);
  EXPECT_NEAR(jac2(0, 1), 3.0, eps);
  EXPECT_NEAR(jac2(1, 0), 1.0, eps);
  EXPECT_NEAR(jac2(1, 1), -4.0, eps);
}

// Test composition of functions
TEST_F(ForwardDifferentiatorTest, FunctionComposition) {
  // Test f(x) = sin(cos(x))
  auto f = [](DualVar<double> x) {
    return sin(cos(x));
  };
  
  double x0 = M_PI/3;
  auto result = derivative(f, x0);
  
  // Expected: f'(x) = cos(cos(x)) * (-sin(x))
  double expected_real = std::sin(std::cos(x0));
  double expected_inf = std::cos(std::cos(x0)) * (-std::sin(x0));
  
  EXPECT_NEAR(result.getReal(), expected_real, eps);
  EXPECT_NEAR(result.getInf(), expected_inf, eps);
}

// Test gradient of multivariate composition
TEST_F(ForwardDifferentiatorTest, MultivariateComposition) {
  // Test f(x,y) = exp(x^2 + y^2)
  auto f = [](std::vector<DualVar<double>> vars) {
    return exp(vars[0] * vars[0] + vars[1] * vars[1]);
  };
  
  std::vector<double> x = {1.0, 2.0};
  auto grad = gradient(f, x);
  
  // Expected: df/dx = 2x*exp(x^2+y^2), df/dy = 2y*exp(x^2+y^2)
  double exp_val = std::exp(5.0);  // exp(1^2 + 2^2)
  
  EXPECT_NEAR(grad[0], 2.0 * exp_val, eps);  // 2*1*exp(5)
  EXPECT_NEAR(grad[1], 4.0 * exp_val, eps);  // 2*2*exp(5)
}

// Test with very small and very large numbers
TEST_F(ForwardDifferentiatorTest, NumericalStability) {
  // Test with very small numbers
  auto f_small = [](DualVar<double> x) { return x * x; };
  auto result_small = derivative(f_small, 1e-10);
  EXPECT_NEAR(result_small.getReal(), 1e-20, 1e-25);
  EXPECT_NEAR(result_small.getInf(), 2e-10, 1e-15);

  // Test with very large numbers
  auto f_large = [](DualVar<double> x) { return x * x; };
  auto result_large = derivative(f_large, 1e5);
  EXPECT_NEAR(result_large.getReal(), 1e10, 1e5);
  EXPECT_NEAR(result_large.getInf(), 2e5, 1e0);
}