// DualVar.h
#ifndef __DUALVAR__H__
#define __DUALVAR__H__

#include <array>
#include <iostream>
#include <cstddef>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <functional>
//#include <Eigen/Dense>
#include <stdexcept>

namespace autodiff {
namespace forward {

template <typename T>
class DualVar {
public:
    DualVar() = default;

    DualVar(T const & real):
        real_{real} {}

    DualVar(T const & real, T const & inf):
        real_{real}, inf_{inf} {}

    std::string getValue() const {
        return "(" + std::to_string(real_) + ", " +
            std::to_string(inf_) + ")";
    }

    T getReal() const { return real_; }
    T getInf() const { return inf_; }

    void setInf(T inf) { inf_ = inf; }

    /***************************************************************/
    /* NEGATE */
    /***************************************************************/
    DualVar<T> operator-() const {
        return DualVar<T>(-real_, -inf_);
    }

    /***************************************************************/
    /* SUM */
    /***************************************************************/
    DualVar<T> operator+(DualVar<T> const & rhs) const {
        return DualVar<T>(real_ + rhs.real_, rhs.inf_ + inf_);
    }

    DualVar<T> operator+(T const & rhs) const {
        return DualVar<T>(real_ + rhs, inf_);
    }

    /***************************************************************/
    /* SUB */
    /***************************************************************/
    DualVar<T> operator-(DualVar<T> const & rhs) const {
        return DualVar<T>(real_ - rhs.real_, inf_ - rhs.inf_);
    }

    DualVar<T> operator-(T const & rhs) const {
        return DualVar<T>(real_ - rhs, inf_);
    }

    /***************************************************************/
    /* MUL */
    /***************************************************************/
    DualVar<T> operator*(DualVar<T> const & rhs) const {
        return DualVar<T>(real_ * rhs.real_,
                real_ * rhs.inf_ + inf_ * rhs.real_);
    }

    DualVar<T> operator*(T const & rhs) const {
        return DualVar<T>(real_ * rhs, rhs * inf_);
    }

    /***************************************************************/
    /* DIV */
    /***************************************************************/
    DualVar<T> operator/(DualVar<T> const & rhs) const {
    return DualVar<T>(real_ / rhs.real_,
        (inf_ * rhs.real_ + real_ * rhs.inf_) / (rhs.real_ * rhs.real_));
    }

    DualVar<T> operator/(T const & rhs) const {
        return DualVar<T>(real_ / rhs, inf_ * rhs / (rhs * rhs));
    }

    template <typename U>
    friend DualVar<U> operator/(U const & lhs, DualVar<U> const & rhs);

    /***************************************************************/
    /* MISC                                                        */
    /***************************************************************/
    template <typename U>
    friend DualVar<U> abs(DualVar<U> const & arg);

    template <typename U>
    friend DualVar<U> cos(DualVar<U> const & arg);

    template <typename U>
    friend DualVar<U> sin(DualVar<U> const & arg);

    template <typename U>
    friend DualVar<U> tan(DualVar<U> const & arg);

    template <typename U>
    friend DualVar<U> log(DualVar<U> const & arg);

    template <typename U>
    friend DualVar<U> exp(DualVar<U> const & arg);

    template <typename U>
    friend DualVar<U> pow(DualVar<U> const & base, DualVar<U> const & exp);

    template <typename U>
    friend DualVar<U> pow(U const & base, DualVar<U> const & exp);

    template <typename U>
    friend DualVar<U> pow(DualVar<U> const & base, U const & exp);

    template <typename U>
    friend DualVar<U> relu(DualVar<U> const & arg);

    bool operator==(DualVar<T> const & rhs) {
        return (real_ == rhs.real_) && (inf_ == rhs.inf_);
    }

private:
    T const real_ = 0;
    T inf_ = 0;
};

} // namespace forward
} // namespace autodiff

#endif // __DUALVAR__H__
