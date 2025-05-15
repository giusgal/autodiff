// DualOps.h
#ifndef __DUALOPS__H__
#define __DUALOPS__H__

#include "DualVar.h"

// Note: Eigen include left commented as in original
//#include <Eigen/Dense>

namespace autodiff
{
    namespace forward
    {
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
        DualVar<T> relu(DualVar<T> const & arg) {
            if (arg.real_ > 0){
                return DualVar<T>(arg.real_, 1);
            } else {
                return DualVar<T>(0, 0);
            }
        }
    }
}
#endif // __DUALOPS__H__
