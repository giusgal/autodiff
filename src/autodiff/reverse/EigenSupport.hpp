#ifndef __EIGENSUPPORT_HPP__
#define __EIGENSUPPORT_HPP__

#include "Var.hpp"
#include <Eigen/Core>

/**
 * This file specializes the NumTraits struct template for the
 * Var<double> type to let Eigen access information
 * on this type
 *
 * Taken from:
 * "https://eigen.tuxfamily.org/dox/TopicCustomizing_CustomScalar.html"
 */

// TODO: maybe it's not a good idea to leave this type-alias here
// (namespace pollution)
using VarD = autodiff::reverse::Var<double>;

namespace Eigen {


// TODO: review this
template<>
struct NumTraits<VarD> : NumTraits<double> {
    /* 
    Real gives the "real part" type of T. If T is already real, 
    then Real is just a typedef to T. If T is std::complex<U> 
    then Real is a typedef to U
    */
    typedef VarD Real;
    /*
    NonInteger gives the type that should be used for operations 
    producing non-integral values, such as quotients, square roots, etc. 
    If T is a floating-point type, then this typedef just gives T again.
    Thus, this typedef is only intended as a helper for code that needs
    to explicitly promote types.
    */
    typedef VarD NonInteger;
    /*
    Nested gives the type to use to nest a value inside of the expression tree.
    */
    typedef VarD Nested;

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


}; // namespace Eigen

namespace autodiff {
namespace reverse {

inline VarD const & conj(VarD const & x) { return x; }
inline VarD const & real(VarD const & x) { return x; }
inline VarD abs2(VarD const & x) { return x*x; }

}; // namespace reverse
}; // namespace autodiff

#endif // __EIGENSUPPORT_HPP__
