#pragma once

#include <functional>
#include <Eigen/Dense>
#include "Jacobian.hpp"

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
    Newton(JacobianBase & J, NewtonOpts opts):
        opts_{opts}, J_{J} {}


    /**
     * Solves the system with initial guess `x0`
     *
     * @param x0 initial guess
     */
    RealVec solve(RealVec const & x0) {
        RealVec x(x0.size());
        RealVec delta(x0.size());
        RealVec resid;

        x = x0;

        size_t iter = 0;
        for(; iter < opts_.maxit; ++iter) {
            delta = J_.solve(x, resid);

            x = x - delta;

            delta = delta.cwiseAbs();
            resid = resid.cwiseAbs();

            double step_size =
                std::accumulate(delta.data(), delta.data() + delta.size(), 0.0);
            double resid_sum =
                std::accumulate(resid.data(), resid.data() + resid.size(), 0.0);

            if((step_size < opts_.tol) && (resid_sum < opts_.tol)) {
                break;
            }
        }

        if (iter == opts_.maxit) {
            std::cout << "Unable to converge" << std::endl;
        } else {
            std::cout << "Converged in " << iter << " iterations." << std::endl;
        }

        return x;
    }
private:
    JacobianBase & J_;
    NewtonOpts opts_;
};

}; // namespace newton
