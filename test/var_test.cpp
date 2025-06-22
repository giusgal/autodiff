#include <gtest/gtest.h>
#include "Var.hpp"

using Var = autodiff::reverse::Var<double>;

/**
 * Unit tests for the functionalities exposed by Var
 */

// ****** CREATION OF NEW OBJECTS ******
TEST(VarTest, TestValueConstructor) {
    Var y = 1024.0;
    EXPECT_EQ(y.value(), 1024.0);
}

TEST(VarTest, TestValueAndGradNewObject) {
    // Not initialized
    Var x;
    EXPECT_EQ(x.grad(), 0.0);
    EXPECT_EQ(x.value(), 0.0);

    // Initialized
    Var y = 1024.0;
    EXPECT_EQ(x.grad(), 0.0);
    EXPECT_EQ(x.value(), 0.0);
}

// ************ OPERATORS ************
// NOTE:
//  Each test (TEST_F) is executed independently from the others.
//  That is, each time a test is executed, new variables are
//  appendened to the Tape.
class VarTestFixture : public testing::Test {
protected:
    Var x = -12.0;
    Var y = 33.0;
    Var z;
};

// operator+
TEST_F(VarTestFixture, TestOperatorSumUnary) {
    z = +x;
    ASSERT_EQ(z.value(), x.value());
    z.backward();
    ASSERT_EQ(x.grad(), 1.0);
}
TEST_F(VarTestFixture, TestOperatorSumVarVar) {
    z = x + y;
    ASSERT_EQ(z.value(), x.value() + y.value());
    z.backward();
    ASSERT_EQ(x.grad(), 1.0);
    ASSERT_EQ(y.grad(), 1.0);
}
TEST_F(VarTestFixture, TestOperatorSumVarScalar) {
    constexpr double scalar = 2;
    z = x + scalar;
    ASSERT_EQ(z.value(), x.value() + scalar);
    z.backward();
    ASSERT_EQ(x.grad(), 1.0);
}
TEST_F(VarTestFixture, TestOperatorSumScalarVar) {
    constexpr double scalar = 2;
    z = scalar + x;
    ASSERT_EQ(z.value(), scalar + x.value());
    z.backward();
    ASSERT_EQ(x.grad(), 1.0);
}
TEST_F(VarTestFixture, TestOperatorSumAssignmentVar) {
    const double old_z = z.value();
    z += x;
    ASSERT_EQ(z.value(), old_z + x.value());
    z.backward();
    ASSERT_EQ(x.grad(), 1.0);
}
TEST_F(VarTestFixture, TestOperatorSumAssignmentScalar) {
    const double old_z = z.value();
    constexpr double scalar = 2;
    z += scalar;
    ASSERT_EQ(z.value(), old_z + scalar);
}

// operator-
TEST_F(VarTestFixture, TestOperatorSubUnary) {
    z = -y;
    ASSERT_EQ(z.value(), -y.value());
    z.backward();
    ASSERT_EQ(y.grad(), -1.0);
}
TEST_F(VarTestFixture, TestOperatorSubVarVar) {
    z = x - y;
    ASSERT_EQ(z.value(), x.value() - y.value());
    z.backward();
    ASSERT_EQ(x.grad(), 1.0);
    ASSERT_EQ(y.grad(), -1.0);
}
TEST_F(VarTestFixture, TestOperatorSubVarScalar) {
    constexpr double scalar = 2;
    z = x - scalar;
    ASSERT_EQ(z.value(), x.value() - scalar);
    z.backward();
    ASSERT_EQ(x.grad(), 1.0);
}
TEST_F(VarTestFixture, TestOperatorSubScalarVar) {
    constexpr double scalar = 2;
    z = scalar - x;
    ASSERT_EQ(z.value(), scalar - x.value());
    z.backward();
    ASSERT_EQ(x.grad(), -1.0);
}
TEST_F(VarTestFixture, TestOperatorSubAssignmentVar) {
    const double old_z = z.value();
    z -= x;
    ASSERT_EQ(z.value(), old_z - x.value());
    z.backward();
    ASSERT_EQ(x.grad(), -1.0);
}
TEST_F(VarTestFixture, TestOperatorSubAssignmentScalar) {
    const double old_z = z.value();
    constexpr double scalar = 2;
    z -= scalar;
    ASSERT_EQ(z.value(), old_z - scalar);
}

// opeartor*
TEST_F(VarTestFixture, TestOperatorMulVarVar) {
    z = x * y;
    ASSERT_EQ(z.value(), x.value() * y.value());
    z.backward();
    ASSERT_EQ(x.grad(), y.value());
    ASSERT_EQ(y.grad(), x.value());
}
TEST_F(VarTestFixture, TestOperatorMulVarScalar) {
    constexpr double scalar = 2;
    z = x * scalar;
    ASSERT_EQ(z.value(), x.value() * scalar);
    z.backward();
    ASSERT_EQ(x.grad(), scalar);
}
TEST_F(VarTestFixture, TestOperatorMulScalarVar) {
    constexpr double scalar = 2;
    z = scalar * x;
    ASSERT_EQ(z.value(), scalar * x.value());
    z.backward();
    ASSERT_EQ(x.grad(), scalar);
}
TEST_F(VarTestFixture, TestOperatorMulAssignmentVar) {
    const double old_z = z.value();
    z *= x;
    ASSERT_EQ(z.value(), old_z * x.value());
    z.backward();
    ASSERT_EQ(x.grad(), old_z);
}
TEST_F(VarTestFixture, TestOperatorMulAssignmentScalar) {
    const double old_z = z.value();
    constexpr double scalar = 2;
    z *= scalar;
    ASSERT_EQ(z.value(), old_z * scalar);
}

// opeartor/
TEST_F(VarTestFixture, TestOperatorDivVarVar) {
    z = x / y;
    ASSERT_EQ(z.value(), x.value() / y.value());
    z.backward();
    ASSERT_EQ(x.grad(), 1.0/y.value());
    ASSERT_EQ(y.grad(), (-x.value())/(y.value()*y.value()));
}
TEST_F(VarTestFixture, TestOperatorDivVarScalar) {
    constexpr double scalar = 2;
    z = x / scalar;
    ASSERT_EQ(z.value(), x.value() / scalar);
    z.backward();
    ASSERT_EQ(x.grad(), 1.0/scalar);
}
TEST_F(VarTestFixture, TestOperatorDivScalarVar) {
    constexpr double scalar = 2;
    z = scalar / x;
    ASSERT_EQ(z.value(), scalar / x.value());
    z.backward();
    ASSERT_EQ(x.grad(), (-scalar) / (x.value()*x.value()));
}
TEST_F(VarTestFixture, TestOperatorDivAssignmentVar) {
    const double old_z = z.value();
    z /= x;
    ASSERT_EQ(z.value(), old_z / x.value());
    z.backward();
    ASSERT_EQ(x.grad(), (-old_z) / (x.value()*x.value()));
}
TEST_F(VarTestFixture, TestOperatorDivAssignmentScalar) {
    const double old_z = z.value();
    constexpr double scalar = 2;
    z /= scalar;
    ASSERT_EQ(z.value(), old_z / scalar);
}

// abs
TEST_F(VarTestFixture, TestAbs) {
    z = abs(x);
    ASSERT_EQ(z.value(), std::abs(x.value()));
    z.backward();
    ASSERT_EQ(x.grad(), x.value()/std::abs(x.value()));
}

// cos
TEST_F(VarTestFixture, TestCos) {
    z = cos(x);
    ASSERT_EQ(z.value(), std::cos(x.value()));
    z.backward();
    ASSERT_EQ(x.grad(), -std::sin(x.value()));
}

// sin
TEST_F(VarTestFixture, TestSin) {
    z = sin(x);
    ASSERT_EQ(z.value(), std::sin(x.value()));
    z.backward();
    ASSERT_EQ(x.grad(), std::cos(x.value()));
}

// tan
TEST_F(VarTestFixture, TestTan) {
    z = tan(x);
    ASSERT_EQ(z.value(), std::tan(x.value()));
    z.backward();
    auto den = std::cos(x.value());
    ASSERT_EQ(x.grad(), 1.0/(den*den));
}

// log
TEST_F(VarTestFixture, TestLog) {
    z = log(y);
    ASSERT_EQ(z.value(), std::log(y.value()));
    z.backward();
    ASSERT_EQ(y.grad(), 1.0/y.value());
}

// relu
TEST_F(VarTestFixture, TestRelu) {
    auto my_relu = [](double x) {
        return (x > 0) ? x : 0;
    };

    auto der_relu = [](double x) {
        return (x > 0) ? 1 : 0;
    };

    Var in1 = -1.0;
    z = relu(in1);
    ASSERT_EQ(z.value(), my_relu(in1.value()));
    z.backward();
    ASSERT_EQ(in1.grad(), der_relu(in1.value()));

    Var in2 = 1.0;
    z = relu(in2);
    ASSERT_EQ(z.value(), my_relu(in2.value()));
    z.backward();
    ASSERT_EQ(in2.grad(), der_relu(in2.value()));
}

// tanh
TEST_F(VarTestFixture, TestTanh) {
    auto der_tanh = [](double x) {
        return 1.0/std::cosh(x);
    };

    z = tanh(x);
    ASSERT_EQ(z.value(), std::tanh(x.value()));
    z.backward();
    auto den = std::cosh(x.value());
    ASSERT_EQ(x.grad(), 1.0/(den*den));
}

// pow
TEST_F(VarTestFixture, TestPowVarVar) {
    z = pow(y,x);
    ASSERT_EQ(z.value(), std::pow(y.value(), x.value()));
    z.backward();
    ASSERT_EQ(y.grad(), x.value()*std::pow(y.value(),x.value()-1));
    ASSERT_EQ(x.grad(), z.value()*std::log(y.value()));
}

TEST_F(VarTestFixture, TestPowVarScalar) {
    constexpr double scalar = 2.0;

    z = pow(y,scalar);
    ASSERT_EQ(z.value(), std::pow(y.value(), scalar));
    z.backward();
    ASSERT_EQ(y.grad(), scalar*std::pow(y.value(),scalar-1));
}

TEST_F(VarTestFixture, TestPowScalarVar) {
    constexpr double scalar = 2.0;

    z = pow(scalar,y);
    ASSERT_EQ(z.value(), std::pow(scalar, y.value()));
    z.backward();
    ASSERT_EQ(y.grad(), z.value()*std::log(scalar));
}

// exp
TEST_F(VarTestFixture, TestExp) {
    z = exp(x);
    ASSERT_EQ(z.value(), std::exp(x.value()));
    z.backward();
    ASSERT_EQ(x.grad(), z.value());
}

// sqrt
TEST_F(VarTestFixture, TestSqrt) {
    z = sqrt(y);
    ASSERT_EQ(z.value(), std::sqrt(y.value()));
    z.backward();
    ASSERT_EQ(y.grad(), 1.0/(2.0*std::sqrt(y.value())));
}

// ******** LOGICAL OPERATORS ********
// operator<
TEST(VarTest, TestLTVarVar) {
    Var x = 1.0;
    Var y = 2.0;
    ASSERT_TRUE(x < y);
}
TEST(VarTest, TestLTVarScalar) {
    Var x = 1.0;
    ASSERT_TRUE(x < 2.0);
}
TEST(VarTest, TestLTScalarVar) {
    Var x = 1.0;
    ASSERT_TRUE(-1.0 < x);
}


// operator>
TEST(VarTest, TestGTVarVar) {
    Var x = 1.0;
    Var y = 2.0;
    ASSERT_TRUE(y > x);
}
TEST(VarTest, TestGTVarScalar) {
    Var x = 1.0;
    ASSERT_TRUE(x > -1.0);
}
TEST(VarTest, TestGTScalarVar) {
    Var x = 1.0;
    ASSERT_TRUE(2.0 > x);
}

// operator==
TEST(VarTest, TestEqVarVar) {
    Var x = 1.0;
    Var y = 1.0;
    ASSERT_TRUE(x == y);
}
TEST(VarTest, TestEqVarScalar) {
    Var x = 1.0;
    ASSERT_TRUE(x == 1.0);
}
TEST(VarTest, TestEqScalarVar) {
    Var x = 1.0;
    ASSERT_TRUE(1.0 == x);
}

// opearator!=
TEST(VarTest, TestNeqVarVar) {
    Var x = 1.0;
    Var y = 2.0;
    ASSERT_TRUE(x != y);
}
TEST(VarTest, TestNeqVarScalar) {
    Var x = 1.0;
    ASSERT_TRUE(x != 2.0);
}
TEST(VarTest, TestNeqScalarVar) {
    Var x = 1.0;
    ASSERT_TRUE(2.0 != x);
}

// operator<=
TEST(VarTest, TestLTEVarVar) {
    Var x = 1.0;
    Var y = 2.0;
    ASSERT_TRUE(x <= y);
}
TEST(VarTest, TestLTEVarScalar) {
    Var x = 1.0;
    ASSERT_TRUE(x <= 2.0);
}
TEST(VarTest, TestLTEScalarVar) {
    Var x = 1.0;
    ASSERT_TRUE(-1.0 <= x);
}

// operator>=
TEST(VarTest, TestGTEVarVar) {
    Var x = 1.0;
    Var y = 2.0;
    ASSERT_TRUE(y >= x);
}
TEST(VarTest, TestGTEVarScalar) {
    Var x = 1.0;
    ASSERT_TRUE(x >= -1.0);
}
TEST(VarTest, TestGTEScalarVar) {
    Var x = 1.0;
    ASSERT_TRUE(2.0 >= x);
}
