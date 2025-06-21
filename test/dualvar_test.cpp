#include <gtest/gtest.h>
#include <cmath>
#include "DualVar.hpp"

using namespace autodiff::forward;

class DualVarTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test values
        x = DualVar<double>(2.0, 1.0);  
        y = DualVar<double>(3.0, 0.0);  
        z = DualVar<double>(4.0, 2.0);  
        eps = 1e-10;
    }

    DualVar<double> x, y, z;
    double eps;
};

TEST_F(DualVarTest, Constructors) {
    DualVar<double> default_dual;
    EXPECT_EQ(default_dual.getReal(), 0.0);
    EXPECT_EQ(default_dual.getInf(), 0.0);

    DualVar<double> real_only(5.0);
    EXPECT_EQ(real_only.getReal(), 5.0);
    EXPECT_EQ(real_only.getInf(), 0.0);

    DualVar<double> full(3.0, 4.0);
    EXPECT_EQ(full.getReal(), 3.0);
    EXPECT_EQ(full.getInf(), 4.0);

    DualVar<double> copy(full);
    EXPECT_EQ(copy.getReal(), 3.0);
    EXPECT_EQ(copy.getInf(), 4.0);
}

TEST_F(DualVarTest, UnaryNegation) {
    auto neg_x = -x;
    EXPECT_EQ(neg_x.getReal(), -2.0);
    EXPECT_EQ(neg_x.getInf(), -1.0);
}

TEST_F(DualVarTest, Addition) {
    // DualVar + DualVar
    auto sum1 = x + y;
    EXPECT_EQ(sum1.getReal(), 5.0);
    EXPECT_EQ(sum1.getInf(), 1.0);

    // DualVar + scalar
    auto sum2 = x + 3.0;
    EXPECT_EQ(sum2.getReal(), 5.0);
    EXPECT_EQ(sum2.getInf(), 1.0);

    // scalar + DualVar
    auto sum3 = 3.0 + x;
    EXPECT_EQ(sum3.getReal(), 5.0);
    EXPECT_EQ(sum3.getInf(), 1.0);
}

// Test subtraction operations
TEST_F(DualVarTest, Subtraction) {
    // DualVar - DualVar
    auto diff1 = x - y;
    EXPECT_EQ(diff1.getReal(), -1.0);
    EXPECT_EQ(diff1.getInf(), 1.0);

    // DualVar - scalar
    auto diff2 = x - 1.0;
    EXPECT_EQ(diff2.getReal(), 1.0);
    EXPECT_EQ(diff2.getInf(), 1.0);

    // scalar - DualVar
    auto diff3 = 5.0 - x;
    EXPECT_EQ(diff3.getReal(), 3.0);
    EXPECT_EQ(diff3.getInf(), -1.0);
}

// Test multiplication operations
TEST_F(DualVarTest, Multiplication) {
    // DualVar * DualVar
    auto prod1 = x * z;  // (2,1) * (4,2) = (8, 2*2 + 1*4) = (8, 8)
    EXPECT_EQ(prod1.getReal(), 8.0);
    EXPECT_EQ(prod1.getInf(), 8.0);

    // DualVar * scalar
    auto prod2 = x * 3.0;
    EXPECT_EQ(prod2.getReal(), 6.0);
    EXPECT_EQ(prod2.getInf(), 3.0);

    // scalar * DualVar
    auto prod3 = 3.0 * x;
    EXPECT_EQ(prod3.getReal(), 6.0);
    EXPECT_EQ(prod3.getInf(), 3.0);
}

// Test division operations
TEST_F(DualVarTest, Division) {
    // DualVar / DualVar
    auto quot1 = x / y;  // (2,1) / (3,0) = (2/3, (1*3 - 2*0)/(3*3)) = (2/3, 1/3)
    EXPECT_NEAR(quot1.getReal(), 2.0/3.0, eps);
    EXPECT_NEAR(quot1.getInf(), 1.0/3.0, eps);

    // DualVar / scalar
    auto quot2 = x / 2.0;
    EXPECT_EQ(quot2.getReal(), 1.0);
    EXPECT_EQ(quot2.getInf(), 0.5);

    // scalar / DualVar
    auto quot3 = 6.0 / y;  // 6 / (3,0) = (2, -6*0/(3*3)) = (2, 0)
    EXPECT_EQ(quot3.getReal(), 2.0);
    EXPECT_EQ(quot3.getInf(), 0.0);
}

// Test trigonometric functions
TEST_F(DualVarTest, TrigonometricFunctions) {
    DualVar<double> angle(M_PI/4, 1.0);  // 45 degrees with derivative 1
    
    // sin
    auto sin_result = sin(angle);
    EXPECT_NEAR(sin_result.getReal(), std::sin(M_PI/4), eps);
    EXPECT_NEAR(sin_result.getInf(), std::cos(M_PI/4), eps);

    // cos
    auto cos_result = cos(angle);
    EXPECT_NEAR(cos_result.getReal(), std::cos(M_PI/4), eps);
    EXPECT_NEAR(cos_result.getInf(), -std::sin(M_PI/4), eps);

    // tan
    auto tan_result = tan(angle);
    EXPECT_NEAR(tan_result.getReal(), std::tan(M_PI/4), eps);
    EXPECT_NEAR(tan_result.getInf(), 1.0/(std::cos(M_PI/4) * std::cos(M_PI/4)), eps);

    // tanh
    auto tanh_result = tanh(angle);
    double tanh_val = std::tanh(M_PI/4);
    EXPECT_NEAR(tanh_result.getReal(), tanh_val, eps);
    EXPECT_NEAR(tanh_result.getInf(), 1.0 - tanh_val * tanh_val, eps);
}

// Test logarithmic and exponential functions
TEST_F(DualVarTest, LogExpFunctions) {
    // log
    auto log_result = log(x);  // log(2,1) = (log(2), 1/2)
    EXPECT_NEAR(log_result.getReal(), std::log(2.0), eps);
    EXPECT_NEAR(log_result.getInf(), 0.5, eps);

    // exp
    auto exp_result = exp(x);  // exp(2,1) = (exp(2), 1*exp(2))
    EXPECT_NEAR(exp_result.getReal(), std::exp(2.0), eps);
    EXPECT_NEAR(exp_result.getInf(), std::exp(2.0), eps);
}

// Test power functions
TEST_F(DualVarTest, PowerFunctions) {
    // DualVar^DualVar
    DualVar<double> base(2.0, 1.0);
    DualVar<double> exponent(3.0, 0.0);
    auto pow_result1 = pow(base, exponent);
    // (2,1)^(3,0) = (8, 8^(1-1) * (2*0*ln(2) + 3*1)) = (8, 1 * 3) = (8, 3)
    EXPECT_NEAR(pow_result1.getReal(), 8.0, eps);
    EXPECT_NEAR(pow_result1.getInf(), 12.0, eps);  // 3 * 2^2 * 1

    // scalar^DualVar
    auto pow_result2 = pow(2.0, DualVar<double>(3.0, 1.0));
    // 2^(3,1) = (2^3, 2^3 * 1 * ln(2)) = (8, 8*ln(2))
    EXPECT_NEAR(pow_result2.getReal(), 8.0, eps);
    EXPECT_NEAR(pow_result2.getInf(), 8.0 * std::log(2.0), eps);

    // DualVar^scalar
    auto pow_result3 = pow(DualVar<double>(2.0, 1.0), 3.0);
    // (2,1)^3 = (8, 2^2 * 3 * 1) = (8, 12)
    EXPECT_NEAR(pow_result3.getReal(), 8.0, eps);
    EXPECT_NEAR(pow_result3.getInf(), 12.0, eps);
}

// Test square root
TEST_F(DualVarTest, SquareRoot) {
    DualVar<double> four(4.0, 1.0);
    auto sqrt_result = sqrt(four);
    // sqrt(4,1) = (2, 1/(2*sqrt(4))) = (2, 1/4)
    EXPECT_NEAR(sqrt_result.getReal(), 2.0, eps);
    EXPECT_NEAR(sqrt_result.getInf(), 0.25, eps);
}

// Test absolute value
TEST_F(DualVarTest, AbsoluteValue) {
    DualVar<double> positive(3.0, 2.0);
    DualVar<double> negative(-3.0, 2.0);

    auto abs_pos = abs(positive);
    EXPECT_EQ(abs_pos.getReal(), 3.0);
    EXPECT_EQ(abs_pos.getInf(), 2.0);

    auto abs_neg = abs(negative);
    EXPECT_EQ(abs_neg.getReal(), 3.0);
    EXPECT_EQ(abs_neg.getInf(), -2.0);
}

// Test ReLU function
TEST_F(DualVarTest, ReLU) {
    DualVar<double> positive(3.0, 2.0);
    DualVar<double> negative(-3.0, 2.0);
    DualVar<double> zero(0.0, 2.0);

    auto relu_pos = relu(positive);
    EXPECT_EQ(relu_pos.getReal(), 3.0);
    EXPECT_EQ(relu_pos.getInf(), 2.0);

    auto relu_neg = relu(negative);
    EXPECT_EQ(relu_neg.getReal(), 0.0);
    EXPECT_EQ(relu_neg.getInf(), 0.0);

    auto relu_zero = relu(zero);
    EXPECT_EQ(relu_zero.getReal(), 0.0);
    EXPECT_EQ(relu_zero.getInf(), 0.0);
}

// Test equality operator
TEST_F(DualVarTest, EqualityOperator) {
    DualVar<double> a(2.0, 1.0);
    DualVar<double> b(2.0, 1.0);
    DualVar<double> c(2.0, 2.0);
    DualVar<double> d(3.0, 1.0);

    EXPECT_TRUE(a == b);
    // we only care about real part
    EXPECT_TRUE(a == c);
    EXPECT_FALSE(a == d);
}

// Test getValue method
TEST_F(DualVarTest, GetValue) {
    DualVar<double> test(3.14, 2.71);
    std::string expected = "(3.140000, 2.710000)";
    EXPECT_EQ(test.getValue(), expected);
}

// Test setInf method
TEST_F(DualVarTest, SetInf) {
    DualVar<double> test(3.0, 1.0);
    test.setInf(5.0);
    EXPECT_EQ(test.getInf(), 5.0);
    EXPECT_EQ(test.getReal(), 3.0);  // Real part should remain unchanged
}

// Test edge cases and special values
TEST_F(DualVarTest, EdgeCases) {
    // Test division by zero handling in division operator
    DualVar<double> zero_real(0.0, 1.0);
    // Note: Division by zero will result in inf/nan, testing for proper handling
    
    // Test very small numbers
    DualVar<double> small(1e-15, 1e-15);
    auto small_result = small + small;
    EXPECT_NEAR(small_result.getReal(), 2e-15, 1e-16);
    EXPECT_NEAR(small_result.getInf(), 2e-15, 1e-16);

    // Test very large numbers
    DualVar<double> large(1e15, 1e15);
    auto large_result = large + large;
    EXPECT_NEAR(large_result.getReal(), 2e15, 1e14);
    EXPECT_NEAR(large_result.getInf(), 2e15, 1e14);
}

// Test function composition
TEST_F(DualVarTest, FunctionComposition) {
    DualVar<double> input(2.0, 1.0);
    
    // Test sin(cos(x))
    auto composed = sin(cos(input));
    
    // Manually compute expected values
    double cos_val = std::cos(2.0);
    double sin_cos_val = std::sin(cos_val);
    double expected_real = sin_cos_val;
    double expected_inf = std::cos(cos_val) * (-std::sin(2.0)) * 1.0;
    
    EXPECT_NEAR(composed.getReal(), expected_real, eps);
    EXPECT_NEAR(composed.getInf(), expected_inf, eps);
}