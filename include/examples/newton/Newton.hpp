#pragma once

#include <functional>
#include <Eigen/Dense>
#include "Jacobian.hpp"

namespace newton
{
struct NewtonOpts {
  int maxit;
  double tol;
  int dim_in;
  int dim_out;
};

class Newton : public newton::JacobianTraits
{
public: 
  Newton(
    JacobianBase & J, NewtonOpts opts
  ):
    _opts{opts}, 
    _J{J} {};

  
  RealVec solve(const RealVec &x0) {
    RealVec x(_opts.dim_in), x1(_opts.dim_in), resid(_opts.dim_in);
    x = x0;

    int iter = 0;
    for(; iter < _opts.maxit; iter++) {
      x1 = _J.solve(x, resid);

      x = x - x1;

      x1 = x1.cwiseAbs();
      resid = resid.cwiseAbs();

      double step_size = std::accumulate(x1.data(), x1.data() + x1.size(), 0.0);
      double resid_sum = std::accumulate(resid.data(), resid.data() + resid.size(), 0.0);


      if (step_size < _opts.tol && resid_sum < _opts.tol) {
        break;
      }
    }
    if (iter == _opts.maxit) {
      std::cout << "Unable to converge" << std::endl;
    } else {
      std::cout << "Converged in " << iter << " iterations." << std::endl;
    }
    return x;
  }


private:

  JacobianBase & _J;
  // non linear system
  NewtonOpts _opts;
};

}; // namespace newton
