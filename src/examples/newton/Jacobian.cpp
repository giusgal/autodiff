#include "Jacobian.hpp"

namespace newton {

ForwardJac::ForwardJac(FwNLSType const & fn): fn_{fn} {}

JacobianTraits::RealVec ForwardJac::solve(const RealVec & x, RealVec & resid) {
    JacType J;
    // update the jacobian
    autodiff::forward::jacobian<double>(fn_, x, resid, J);
    return J.fullPivLu().solve(resid);
}

ReverseJac::ReverseJac(RvNLSType const & fn): fn_{fn} {}

JacobianTraits::RealVec ReverseJac::solve(const RealVec & x, RealVec & resid) {
    JacType J;
    // update the jacobian
    autodiff::reverse::jacobian(fn_, x, resid, J);
    return J.fullPivLu().solve(resid);
}

#ifdef __CUDACC__
CudaJac::CudaJac(CudaFunctionWrapper<double> cuda_fn): cuda_fn_(cuda_fn) {}

JacobianTraits::RealVec CudaJac::solve(const RealVec &x, RealVec &resid) {
    JacType J;
    // update the jacobian
    autodiff::forward::jacobian_cuda<double>(cuda_fn_, x, resid, J, eval=1);
    return J.fullPivLu().solve(resid);
}
#endif // __CUDACC__

} // namespace newton