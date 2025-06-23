#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <functional>
#include <Eigen/Dense>
#include "Newton.hpp"

using namespace newton;

using DualVar = autodiff::forward::DualVar<double>;
using DualVec = autodiff::forward::DualVec<double>;
using RealVec = Eigen::VectorXd;

using Var = autodiff::reverse::Var<double>;
using VarVec = Eigen::Matrix<Var, Eigen::Dynamic, 1>;

// taken from https://en.wikipedia.org/wiki/Newton%27s_method
DualVec forward_fn(const DualVec &x)
{
  DualVec res(2);
  res << 5.0 * x(0) * x(0) + x(1) * x(1) * x(0) + sin(2.0 * x(1)) * sin(2.0 * x(1)) - 2.0,
    autodiff::forward::exp(2.0 * x(0) - x(1)) + 4.0 * x(1) - 3.0;

  return res;
}

VarVec rev_fn(VarVec const & x) {
    VarVec res(2);

    res <<
        (
         5.0 * x(0) * x(0) + x(1) * x(1) * x(0) +
         sin(2.0 * x(1)) * sin(2.0 * x(1)) - 2.0
        ),
        (
         pow(std::exp(1), 2.0 * x(0)-x(1)) +
         4.0 * x(1) - 3.0
        );

    return res;
}


class NewtonTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    eps = 1e-6;
    rf = rev_fn;
    ff = forward_fn;
    // initial guess
    x0.resize(2);
    x0 << 1.0, 1.0;
  }

  RealVec x0;
  double eps;
  ForwardJac::FwNLSType ff;
  ReverseJac::RvNLSType rf;
};

TEST_F(NewtonTest, newtontest_1it)
{
  NewtonOpts newtonopts = {
    .maxit = 1,
    .tol = 1e-6
  };

  ForwardJac J_f(ff);
  Newton fwsolver(J_f, newtonopts);
  auto fwres = fwsolver.solve(x0);

  EXPECT_NEAR(fwres[0], 0.617789, eps);
  EXPECT_NEAR(fwres[1], -0.279818, eps); 

  ReverseJac J_r(rf);
  Newton revsolver(J_r, newtonopts);
  auto revres = revsolver.solve(x0);

  EXPECT_NEAR(revres[0], 0.617789, eps);
  EXPECT_NEAR(revres[1], -0.279818, eps);
}


TEST_F(NewtonTest, newtontest_2it)
{
  NewtonOpts newtonopts = {
    .maxit = 2,
    .tol = 1e-6
  };

  ForwardJac J_f(ff);
  Newton fwsolver(J_f, newtonopts);
  auto fwres = fwsolver.solve(x0);

  EXPECT_NEAR(fwres[0], 0.568334, eps);
  EXPECT_NEAR(fwres[1], -0.312859, eps);

  ReverseJac J_r(rf);
  Newton revsolver(J_r, newtonopts);
  auto revres = revsolver.solve(x0);

  EXPECT_NEAR(revres[0], 0.568334, eps);
  EXPECT_NEAR(revres[1], -0.312859, eps);
}


TEST_F(NewtonTest, newtontest_3it)
{
  NewtonOpts newtonopts = {
    .maxit = 3,
    .tol = 1e-6
  };

  ForwardJac J_f(ff);
  Newton fwsolver(J_f, newtonopts);
  auto fwres = fwsolver.solve(x0);

  EXPECT_NEAR(fwres[0], 0.567305, eps);
  EXPECT_NEAR(fwres[1], -0.309435, eps);

  ReverseJac J_r(rf);
  Newton revsolver(J_r, newtonopts);
  auto revres = revsolver.solve(x0);

  EXPECT_NEAR(revres[0], 0.567305, eps);
  EXPECT_NEAR(revres[1], -0.309435, eps);
}


TEST_F(NewtonTest, newtontest_4it)
{
  NewtonOpts newtonopts = {
    .maxit = 4,
    .tol = 1e-6
  };

  ForwardJac J_f(ff);
  Newton fwsolver(J_f, newtonopts);
  auto fwres = fwsolver.solve(x0);

  EXPECT_NEAR(fwres[0], 0.567297, eps);
  EXPECT_NEAR(fwres[1], -0.309442, eps);

  ReverseJac J_r(rf);
  Newton revsolver(J_r, newtonopts);
  auto revres = revsolver.solve(x0);

  EXPECT_NEAR(revres[0], 0.567297, eps);
  EXPECT_NEAR(revres[1], -0.309442, eps);
}

// expect convergence in 4 iterations
TEST_F(NewtonTest, newtontest_5it)
{
  NewtonOpts newtonopts = {
    .maxit = 5,
    .tol = 1e-6
  };

  ForwardJac J_f(ff);
  Newton fwsolver(J_f, newtonopts);
  auto fwres = fwsolver.solve(x0);

  EXPECT_NEAR(fwres[0], 0.567297, eps);
  EXPECT_NEAR(fwres[1], -0.309442, eps);

  ReverseJac J_r(rf);
  Newton revsolver(J_r, newtonopts);
  auto revres = revsolver.solve(x0);

  EXPECT_NEAR(revres[0], 0.567297, eps);
  EXPECT_NEAR(revres[1], -0.309442, eps);
}




