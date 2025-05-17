#ifndef __DUALVAR__HPP__
#define __DUALVAR__HPP__

#include <array>
#include <iostream>
#include <cstddef>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <functional>
#include <stdexcept>

namespace autodiff {
namespace forward {

template <typename T>
class DualVar {
public:
    DualVar() = default;
    
    // copy constructor
    DualVar(const DualVar<T> &dv) = default;

    DualVar(T const & real):
        real_{real} {}

    DualVar(T const & real, T const & inf):
        real_{real}, inf_{inf} {}

    ~DualVar() = default;

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
    friend DualVar<U> sqrt(DualVar<U> const & arg);

    template <typename U>
    friend DualVar<U> relu(DualVar<U> const & arg);

    template <typename U>
    friend DualVar<U> tanh(DualVar<U> const & arg);


    bool operator==(DualVar<T> const & rhs) {
        return (real_ == rhs.real_) && (inf_ == rhs.inf_);
    }

private:
    T real_ = 0;
    T inf_ = 0;
};

/***************************************************************/
/* OSTREAM */
/***************************************************************/

template <typename T>
std::ostream& operator<<(std::ostream& os, const DualVar<T>& dv) {
    os << "(" << dv.getReal() << ", " << dv.getInf() << ")";
    return os;
}


/***************************************************************/
/* SUM */
/***************************************************************/
template <typename T>
DualVar<T> operator+(T const & lhs, DualVar<T> const & rhs) {
    return rhs+lhs;
}


/***************************************************************/
/* SUB */
/***************************************************************/
template <typename T>
DualVar<T> operator-(T const & lhs, DualVar<T> const & rhs) {
    return -(rhs-lhs);
}

/***************************************************************/
/* MUL */
/***************************************************************/
template <typename T>
DualVar<T> operator*(T const & lhs, DualVar<T> const & rhs) {
    return rhs*lhs;
}

/***************************************************************/
/* DIV */
/***************************************************************/
template <typename T>
DualVar<T> operator/(T const & lhs, DualVar<T> const & rhs) {
    return DualVar<T> (lhs / rhs.real_, lhs * rhs.inf_ / (rhs.real_ * rhs.real_));
}

/***************************************************************/
/* MISC                                                        */
/***************************************************************/
template <typename T>
DualVar<T> abs(DualVar<T> const & arg) {
    int sign_real = (arg.real_ >= 0) ? 1 : -1;
    return DualVar<T> (std::abs(arg.real_), arg.inf_ * sign_real);
}

template <typename T>
DualVar<T> cos(DualVar<T> const & arg) {
    return DualVar<T> (std::cos(arg.real_)
            - arg.inf_ * std::sin(arg.real_));
}

template <typename T>
DualVar<T> sin(DualVar<T> const & arg) {
    return DualVar<T> (std::sin(arg.real_),
            arg.inf_ * std::cos(arg.real_));
}

template <typename T>
DualVar<T> tan(DualVar<T> const & arg) {
    return DualVar<T> (std::tan(arg.real_), 
        arg.inf_ / (std::cos(arg.real_) * std::cos(arg.real_)));
}

template <typename T>
DualVar<T> log(DualVar<T> const & arg) {
    return DualVar<T>(std::log(arg.real_), arg.inf_ / arg.real_);
}

template <typename T>
DualVar<T> exp(DualVar<T> const & arg) {
    return DualVar<T>(std::exp(arg.real_), arg.inf_*std::exp(arg.real_));
}

/* When raising a dual number to the power of another dual number, you get
   (a+b𝜀)^(c+d𝜀) = a^c + a^(c-1)*(a*d*ln(a) + c*b)𝜀                         */
template <typename T>
DualVar<T> pow(DualVar<T> const & base, DualVar<T> const & exp) {
    return DualVar<T>(
        std::pow(
            base.real_,
            exp.real_
        ),
        std::pow(
            base.real_,
            exp.real_ - 1) * (base.real_ * exp.inf_
                * std::log(base.real_) + exp.real_ * base.inf_
        )
    );
}

template <typename T>
DualVar<T> pow(T const & base, DualVar<T> const & exp) {
    return DualVar<T>(
        std::pow(base, exp.real_),
        std::pow(base, exp.real_) * exp.inf_ * std::log(base)
    );
}

template <typename T>
DualVar<T> pow(DualVar<T> const & base, T const & exp) {
    return DualVar<T>(
        std::pow(base.real_, exp), 
        std::pow(base.real_, exp - 1) * exp * base.inf_
    );
}

template <typename T>
DualVar<T> sqrt(DualVar<T> const & arg) {
    return DualVar<T>(
        std::sqrt(arg.real_),
        (1.0/(2.0*std::sqrt(arg.real_)))*arg.inf_
    );
}

template <typename T>
DualVar<T> relu(DualVar<T> const & arg) {
    if (arg.real_ > 0){
        return DualVar<T>(arg.real_, 1);
    } else {
        return DualVar<T>(0, 0);
    }
}

    template <typename T>
    DualVar<T> tanh(DualVar<T> const & arg)
{
    T val = std::tanh(arg.getReal());
    T deriv = 1.0 - val * val;
    return DualVar<T>(val, deriv * arg.inf_);

}

DualVar<double> derivative(std::function<DualVar<double>(DualVar<double>)>f, double x0){
    DualVar<double> res = f(DualVar<double>(x0, 1.0));
    return res;
}

std::vector<double> gradient(std::function<DualVar<double>(std::vector<DualVar<double>>)>f, 
    std::vector<double> x) {

        std::vector<DualVar<double>> xd;
        std::vector<double> res;
        
        xd.reserve(size(x));
        res.reserve(size(x));

        for(int i = 0; i < size(x); i++){
            xd.push_back(DualVar<double>(x[i], 0.0));
        }

        for(int i = 0; i < size(x); i++){
            xd[i].setInf(1.0);
            res.push_back(f(xd).getInf());
            xd[i].setInf(0.0);
        }
        
        return res;
}

    
}; // namespace forward
}; // namespace autodiff

#endif // __DUALVAR__HPP__
