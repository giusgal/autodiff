#pragma once

#include "DualVar.hpp"
#include <Eigen/Core>


using dv = autodiff::forward::DualVar<double>;

/**
 * This file specializes the NumTraits struct template for the
 * DualVar<T> type to let Eigen access information
 * on this type
 *
 * Taken from:
 * "https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html"
 */

namespace Eigen {
template<>
struct NumTraits<dv>
  : NumTraits<double>
{
  /* 
    Real gives the "real part" type of T. If T is already real, 
    then Real is just a typedef to T. If T is std::complex<U> 
    then Real is a typedef to U
  */
  typedef dv Real;
  /*
    NonInteger gives the type that should be used for operations 
    producing non-integral values, such as quotients, square roots, etc. 
    If T is a floating-point type, then this typedef just gives T again.
    Thus, this typedef is only intended as a helper for code that needs
    to explicitly promote types.
  */
  typedef dv NonInteger;
  /*
    Nested gives the type to use to nest a value inside of the expression tree.
  */
  typedef dv Nested;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 2,
    AddCost = 3,
    MulCost = 3
  };
};

inline const dv& conj(const dv &x) { return x; }
inline const dv& real(const dv &x) { return x; }
inline dv abs2(const dv &x) { return x*x; }

}
