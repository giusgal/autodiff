#ifndef __AUTODIFF__H__
#define __AUTODIFF__H__

#include <array>
#include <iostream>
#include <cstddef>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <functional>
#include <Eigen/Dense>
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

Eigen::MatrixXd jacobian(std::function<std::vector<DualVar<double>>(std::vector<DualVar<double>>)>f, Eigen::VectorXd x0, Eigen::VectorXd &eval) {

    std::vector<DualVar<double>> inputs, res;
    int M = eval.size();
    int N = x0.size();
    inputs.reserve(N);

    // Initialize inputs with the values from x0 and seed value set to zero
    for (int i = 0; i < N; i ++) {
        inputs.emplace_back(DualVar<double>(x0[i], 0.0));
    }

    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(M, N);

    // Compute each column of the Jacobian
    // For each column of the jacobian, we are evaluating the function at point x0
    for (int i = 0; i < N; i++) {
        inputs[i].setInf(1.0);
        res = f(inputs);
        for (int j = 0; j < M; j++) {
            jacobian(j, i) = res[j].getInf();
        }
        inputs[i].setInf(0.0);
    }

    // recycle the last function evaluation
    for (int i = 0; i < M; i++) {
        eval[i] = res[i].getReal();
    }
        

    return jacobian;
}

Eigen::VectorXd solve(std::function<std::vector<DualVar<double>>(std::vector<DualVar<double>>)>f, 
                        Eigen::VectorXd x0, int M, Eigen::VectorXd &f_eval) {
        
        // Matrix containing all partial derivatives of function f at point x = x0
        Eigen::MatrixXd J = jacobian(f, x0, f_eval);
        
        for (auto xi : f_eval) {
            if (std::isinf(xi) or std::isnan(xi)) throw std::overflow_error("function valuation is NaN");
        }

        for (auto xi : f_eval) {
            assert(not(std::isnan(xi)));
        }
        // Solve the linear system J * u = f_eval for u
        return J.fullPivLu().solve(f_eval);
}

Eigen::VectorXd newton(std::function<std::vector<DualVar<double>>(std::vector<DualVar<double>>)>f, 
                        Eigen::VectorXd x0, int M, int maxit=1000, double tol=1e-6, bool v = false){

        Eigen::VectorXd x, x1, eval(M);
        x = x0;
        int i = 0;
        for(; i < maxit; i++){
            try{
                x1 = solve(f, x, M, eval);
            }
            catch (const std::overflow_error & e){
                std::cout << e.what() << std::endl;
                i = maxit;
                break;
            }
            x = x - x1;
            if (v) {
                std::cout << "iteration: " << i << std::endl;
                std::cout << "new guess:\n" << x << std::endl;
            }
            x1 = x1.cwiseAbs();
            eval = eval.cwiseAbs();
            double step_size = std::accumulate(x1.data(), x1.data() + x1.size(), 0.0);
            double residual = std::accumulate(eval.data(), eval.data() + eval.size(), 0.0);
            if (step_size < tol && residual < tol)
                break;
        }
        if (i == maxit) {
            throw std::runtime_error("Unable to converge");
        }
        if (v) {
            std::cout << "number of iterations: " << i << std::endl; 
            std::cout << "x:\n" << x << std::endl;
        }
        for (auto xi : eval) {
            assert(not (std::isnan(xi) || std::isinf(xi)));
        }
        // double check
        std::vector<DualVar<double>> x_d(x.size()), x_d1(x.size());
        for(auto xi : x) {
            x_d.push_back(xi);
        }
        auto res_c = f(x_d);
        auto x2 = x + x1;

        for(auto xi : x2) {
            x_d1.push_back(xi);
        }
        auto res_c2 = f(x_d1);
        return x;
    }
    
}; // namespace forward
}; // namespace autodiff

#endif // __AUTODIFF__H__
