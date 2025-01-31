#ifndef __AUTODIFF__H__
#define __AUTODIFF__H__

#include <array>
#include <iostream>
#include <cstddef>
#include <string>
#include <vector>
#include <cmath>

namespace autodiff {

template <typename T> class Tape;
template <typename T> class Var;
template <typename T> class DualVar;

template <typename T>
class DualVar {
public:
    DualVar() = default;
    
    DualVar(
        T const & _real,
        T const & _inf
    ):
        real{_real},
        inf{_inf} {}

    std::string getValue() const { return "(" + std::to_string(real) + ", " + 
                                        std::to_string(inf) + ")"; }
    T getReal() const { return real; }
    T getInf() const { return inf; }

private:
    T const real = 0;
    T const inf = 0;
};

/***************************************************************/
/* SUM */
/***************************************************************/

template <typename T>
DualVar<T> operator+(DualVar<T> const & lhs, DualVar<T> const & rhs){
    return DualVar<T>(lhs.getReal() + rhs.getReal(), rhs.getInf() + lhs.getInf());
}

template <typename T>
DualVar<T> operator+(DualVar<T> const & lhs, T const & rhs){
    return DualVar<T> (lhs.getReal() + rhs, lhs.getInf());
}

template <typename T>
DualVar<T> operator+(T const & lhs, DualVar<T> const & rhs){
    return DualVar<T> (lhs + rhs.getReal(), rhs.getInf());
}


/***************************************************************/
/* SUB */
/***************************************************************/

template <typename T>
DualVar<T> operator-(DualVar<T> const & lhs, DualVar<T> const & rhs){
    return DualVar<T>(lhs.getReal() - rhs.getReal(), lhs.getInf() - rhs.getInf());
}

template <typename T>
DualVar<T> operator-(DualVar<T> const & lhs, T const & rhs){
    return DualVar<T> (lhs.getReal() - rhs, lhs.getInf());
}

template <typename T>
DualVar<T> operator-(T const & lhs, DualVar<T> const & rhs){
    return DualVar<T> (lhs - rhs.getReal(), -rhs.getInf());
}

/***************************************************************/
/* MUL */
/***************************************************************/

template <typename T>
DualVar<T> operator*(DualVar<T> const & lhs, DualVar<T> const & rhs){
    return DualVar<T>(lhs.getReal() * rhs.getReal(), 
        lhs.getReal() * rhs.getInf() + lhs.getInf() * rhs.getReal());
}

template <typename T>
DualVar<T> operator*(DualVar<T> const & lhs, T const & rhs){
    return DualVar<T> (lhs.getReal() * rhs, rhs * lhs.getInf());
}

template <typename T>
DualVar<T> operator*(T const & lhs, DualVar<T> const & rhs){
    return DualVar<T> (lhs * rhs.getReal(), lhs * rhs.getInf());
}

/***************************************************************/
/* DIV */
/***************************************************************/

template <typename T>
DualVar<T> operator/(DualVar<T> const & lhs, DualVar<T> const & rhs){
    return DualVar<T>(lhs.getReal() / rhs.getReal(), 
        (lhs.getInf() * rhs.getReal() + lhs.getReal() * rhs.getInf()) / (rhs.getReal() * rhs.getReal()));
}

template <typename T>
DualVar<T> operator/(DualVar<T> const & lhs, T const & rhs){
    return DualVar<T> (lhs.getReal() / rhs, lhs.getInf() * rhs / (rhs * rhs));
}

template <typename T>
DualVar<T> operator/(T const & lhs, DualVar<T> const & rhs){
    return DualVar<T> (lhs / rhs.getReal(), lhs * rhs.getInf() / (rhs.getReal() * rhs.getReal()));
}

/***************************************************************/
/* MISC                                                        */
/***************************************************************/

template <typename T>
inline DualVar<T> abs(DualVar<T> const & arg){
    int sign_real = (arg.getReal() > 0) ? 1 : ((arg.getReal() < 0) ? -1 : 0);
    return DualVar<T> (std::abs(arg.getReal()), arg.getInf() * sign_real);
}

template <typename T>
inline DualVar<T> cos(DualVar<T> const & arg){
    return DualVar<T> (std::cos(arg.getReal()), - arg.getInf() * std::sin(arg.getReal()));
}

template <typename T>
inline DualVar<T> sin(DualVar<T> const & arg){
    return DualVar<T> (std::sin(arg.getReal()), arg.getInf() * std::cos(arg.getReal()));
}

template <typename T>
inline DualVar<T> tan(DualVar<T> const & arg){
    return DualVar<T> (std::tan(arg.getReal()), 
        arg.getInf() / (std::cos(arg.getReal()) * std::cos(arg.getReal())));
}

template <typename T>
inline DualVar<T> log(DualVar<T> const & arg){
    return DualVar<T> (std::log(arg.getReal()), arg.getInf() / arg.getReal());
}


/* When raising a dual number to the power of another dual number, you get
   (a+b𝜀)^(c+d𝜀) = a^c + a^(c-1)*(a*d*ln(a) + c*b)𝜀                         */
template <typename T>
inline DualVar<T> pow(DualVar<T> const & base, DualVar<T> const & exp){
    return DualVar<T> (std::pow(base.getReal(), exp.getReal()), std::pow(base.getReal(), exp.getReal() - 1) * 
        (base.getReal() * exp.getInf() * std::log(base.getReal()) + exp.getReal() * base.getInf()));
}

template <typename T>
inline DualVar<T> pow(T const & base, DualVar<T> const & exp){
    return DualVar<T> (std::pow(base, exp.getReal()), 
        std::pow(base, exp.getReal()) * exp.getInf() * std::log(base));
}

template <typename T>
inline DualVar<T> pow(DualVar<T> const & base, T const & exp){
    return DualVar<T> (std::pow(base.getReal(), exp), 
        std::pow(base.getReal(), exp - 1) * exp * base.getInf());
}

template <typename T>
bool operator==(DualVar<T> const & lhs, DualVar<T> const & rhs){
    if (lhs.getReal() == rhs.getReal() and lhs.getInf() == rhs.getInf()){ return true; }
    return false;
}

template <typename T>
DualVar<T> relu(DualVar<T> const & arg){
    if (arg.getReal() > 0){
        return DualVar<T> (arg.getReal(), 1);
    } else {
        return DualVar<T> (0, 0);
    }
}

}


#endif // __AUTODIFF__H__
