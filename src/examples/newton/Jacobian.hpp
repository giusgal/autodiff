#include <functional>
#include <Eigen/Dense>
#include "NewtonTraits.hpp"
#include <omp.h>


namespace newton 
{

template <typename T>
class JacobianBase : public NewtonTraits<T>
{
public:
  JacobianBase(
    std::size_t M, std::size_t N, 
    const typename JacobianBase::NLSType &fn
  ): _fn{fn}, _M{M}, _N{N} {
  };
  // initialize matrix?

  typename JacobianBase::JacType compute();

  typename JacobianBase::RealVec solve();

protected:
  const typename JacobianBase::NLSType _fn;
  std::size_t _M, _N;
  typename JacobianBase::JacType _J;

};


template <typename T>
class ForwardJac final : 
public JacobianBase<T>
{
  using dv = typename ForwardJac::dv;
  using FwArgType = typename ForwardJac::FwArgType;
  using FwRetType = typename ForwardJac::FwRetType;
  using NLSType = typename ForwardJac::NLSType;
  using JacType = typename ForwardJac::JacType;
  using RealVec = typename ForwardJac::RealVec;

public:
  ForwardJac(
    std::size_t M, std::size_t N, const NLSType &fn
  ): JacobianBase<T>(M, N, fn) {
    this->_J = JacType(M, N);
  };


  void compute(const RealVec &x0, RealVec &real_eval) {

    // create dual vector to feed the function as input
    FwArgType x0d(this->_M);
    FwRetType eval(this->_M);
    for(int i = 0; i < this->_M; i++) {
      x0d[i] = dv(x0[i], 0.0);
    }

    for (int i = 0; i < this->_N; i++) {
      x0d[i].setInf(1.0);
      eval = this->_fn(x0d);
      for (int j = 0; j < this->_M; j++) {
        this->_J(j, i) = eval[j].getInf();
      }
      x0d[i].setInf(0.0);
    }
    // write the value of fn in real_eval pointer
    
    for (int i = 0; i < this->_M; i++) {
      real_eval[i] = eval[i].getReal();
    }
  
  }

  void compute_parallel(const RealVec &x0, RealVec &real_eval) {
    // create dual vector to feed the function as input
    FwArgType x0d(this->_M);
    FwRetType eval(this->_M);
    for(int i = 0; i < this->_M; i++) {
      x0d[i] = dv(x0[i], 0.0);
    }

    #pragma omp parallel for \
      firstprivate(x0d) lastprivate(eval) shared(this->_J)
    for (int i = 0; i < this->_N; i++) {
      x0d[i].setInf(1.0);
      eval = this->_fn(x0d);
      for (int j = 0; j < this->_M; j++) {
        this->_J(j, i) = eval[j].getInf();
      }
      x0d[i].setInf(0.0);
    }
    // write the value of fn in real_eval pointer
    
    for (int i = 0; i < this->_M; i++) {
      real_eval[i] = eval[i].getReal();
    }
  
  }

  RealVec solve(const RealVec &x, RealVec &resid, int parallel=0) {

     // update the jacobian
    if (parallel) {
      compute_parallel(x, resid);
    } else {
      compute(x, resid);
    }
    return this->_J.fullPivLu().solve(resid);
  }

  JacType getJacobian() {
    return this->_J;
  }


};


// template <typename T>
// class ManualJac : public JacobianBase<T>
// {
// public:
//   ManualJac(
//     fn_type<T> &_jacfn, _M, _N
//   ):
//   jacfn{_jacfn}, M{_M}, N{_N} {};

//   Eigen::MatrixXd compute(fn_io_type<T> &x0) {
//     return jacfn(x0);
//   }


// };


//TODO

// template <typename T>
// class ReverseJac : public JacobianBase<Var<T>> {

// }


} // namespace newton