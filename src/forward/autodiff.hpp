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

template <typename T> class Tape;
template <typename T> class Var;
template <typename T> class DualVar;

template <typename T>
class DualVar {
public:
    DualVar() = default;
    
    DualVar(
        T const & _real
    ):
        real{_real} {}

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

    void setInf(T _inf) { inf = _inf; }

private:
    T const real = 0;
    T inf = 0;
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
    int sign_real = (arg.getReal() >= 0) ? 1 : ((arg.getReal() < 0) ? -1 : 0);
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

//TODO: exp function
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

autodiff::DualVar<double> derivative(std::function<autodiff::DualVar<double>(autodiff::DualVar<double>)>f, double x0){
    autodiff::DualVar<double> res = f(DualVar<double>(x0, 1.0));
    return res;
}

std::vector<double> gradient(std::function<autodiff::DualVar<double>(std::vector<autodiff::DualVar<double>>)>f, 
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

Eigen::MatrixXd jacobian(std::function<std::vector<DualVar<double>>(std::vector<DualVar<double>>)>f, 
                        Eigen::VectorXd x0, Eigen::VectorXd &eval) {
    
    
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
    
}


#endif // __AUTODIFF__H__
