#pragma once

#include <functional>
#include <Eigen/Dense>
#include <omp.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "JacobianTraits.hpp"
#include "ForwardEigenSupport.hpp"
#include "CudaSupport.hpp"
#include "ForwardUtility.hpp"

#include "ReverseEigenSupport.hpp"
#include "ReverseUtility.hpp"

namespace newton {

/**
 * @class JacobianBase
 * @brief An abstract class which represents a generic jacobian
 */
class JacobianBase : public JacobianTraits {
public:
    virtual ~JacobianBase() = default;

    virtual RealVec solve(const RealVec &, RealVec &) = 0;
};

/**
 * @class ForwardJacobian 
 * @brief ForwardJacobian allows to solve systems involving
 *  the jacobian of a given non-linear function using 
 *  forward-mode automatic differtentiation
 */
class ForwardJac final : public JacobianBase {
public:
    ForwardJac(FwNLSType const & fn);

    RealVec solve(const RealVec & x, RealVec & resid) override;

protected:
    FwNLSType const & fn_;
};

/**
 * @class ReverseJac
 * @brief ReverseJac allows to solve systems involving
 *  the jacobian of a given non-linear function using
 *  reverse-mode automatic differtentiation
 */
class ReverseJac final : public JacobianBase {
public:
    ReverseJac(RvNLSType const & fn);

    RealVec solve(const RealVec & x, RealVec & resid) override;

protected:
    RvNLSType const & fn_;
};

#ifdef __CUDACC__
class CudaJac final : public JacobianBase {
public:
    CudaJac(CudaFunctionWrapper<double> cuda_fn);

    RealVec solve(const RealVec &x, RealVec &resid) override;

protected:
    CudaFunctionWrapper<double> cuda_fn_;
};
#endif // __CUDACC__

} // namespace newton