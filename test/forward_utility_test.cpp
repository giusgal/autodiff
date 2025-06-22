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
        x = DualVar<double>(3.0, 1.0); 
        eps = 1e-10;
    }

    DualVar<double> x, y, z;
    double eps;
};

template <typename T>
DualVar<T> poly_test(DualVar<T> x) {
  return 12.0 * x * x + 3.0 * x + 4;
}

TEST_F(fwdiff, simple) {
    // DualVar / DualVar
    
    DualVar<double> res = poly_test(x);
    EXPECT_EQ(res.getReal(), 121.0);
    EXPECT_EQ(res.getInf(), 75.0);
}