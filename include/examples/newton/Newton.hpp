#pragma once

#include <functional>
#include <Eigen/Dense>
#include "Jacobian.hpp"  // Changed from .cpp to .hpp

namespace newton {

/**
 * @class NewtonOpts
 * @brief Options for the newton method
 */
struct NewtonOpts {
  size_t maxit;
  double tol;
};

/**
 * @class Newton
 * @brief A class that implements a solver for non-linear
 *  systems of equations
 */
class Newton : public newton::JacobianTraits {
public:
    /**
     * Builds a `Newton` object from a `JacobianBase` and
     * a `NewtonOpts`
     *
     * @param J An instance of `JacobianBase`
     * @param opts Options for the newton method
     */
    Newton(JacobianBase & J, NewtonOpts opts);

    /**
     * Solves the system with initial guess `x0`
     *
     * @param x0 initial guess
     */
    RealVec solve(RealVec const & x0);

private:
    JacobianBase & J_;
    NewtonOpts opts_;
};

}; // namespace newton