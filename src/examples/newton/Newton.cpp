#include "Newton.hpp"
#include <iostream>
#include <numeric>

namespace newton {

Newton::Newton(JacobianBase & J, NewtonOpts opts)
    : opts_{opts}, J_{J} {}

ForwardJac::RealVec Newton::solve(RealVec const & x0) {
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

}; // namespace newton