#ifndef __AUTODIFF__H__
#define __AUTODIFF__H__

#include <array>
#include <iostream>
#include <cstddef>
#include <string>
#include <vector>

namespace autodiff {

template <typename T> class Tape;
template <typename T> class Var;
template <typename T> class DualVar;

enum Operation {
    NOP,
    SUM,
    SUB,
    MUL,
    DIV
};

std::string getStrFromOp(Operation const & op) {
    switch(op) {
        case Operation::SUM:
            return "+";
            break;
        case Operation::SUB:
            return "-";
            break;
        case Operation::MUL:
            return "*";
            break;
        case Operation::DIV:
            return "/";
            break;
        default:
            return "NOP";
            break;
    }
}

// Singleton Tape
template <typename T>
class Tape {
public:
    static Tape<T> & getTape() {
        static Tape<T> instance;
        return instance;
    }

    Var<T> const & push_var(Var<T> const & var) {
        tape.emplace_back(var);
        tape.back().setIdx(tape.size()-1);
        return tape.back();
    }

    size_t size() const {
        return tape.size();
    } 

    Var<T> & operator[](size_t idx) {
        return tape[idx];
    }

private:
    // Disable some predefined operator/constructors
    Tape(Tape<T> const &) = delete;
    void operator=(Tape<T> const &) = delete;

    Tape() {
        // Push a dummy variable as the first element
        push_var(Var<T>{});
    }

    std::vector<Var<T>> tape;
};

template <typename T>
class Var {
public:
    Var() = default;
    
    Var(
        T const & _value,
        size_t const & _left,
        size_t const & _right,
        Operation const & _op
    ):
        value{_value},
        left{_left},
        right{_right},
        op{_op} {}

    Var(T const & _value): value{_value} {
        addToTape();
    }

    T getValue() const { return value; }
    size_t getLeft() const { return left; }
    size_t getRight() const { return right; }
    size_t getIdx() const { return idx; }
    Operation getOperation() const { return op; }

    void setIdx(size_t const & _idx) { idx = _idx; }


private:
    T const value = 0;
    size_t const left = 0;
    size_t const right = 0;
    size_t idx = 0;
    Operation const op = Operation::NOP;

    void addToTape() {
        auto & tape = Tape<T>::getTape();
        auto pushedVar = tape.push_var(*this);
        // Modify the idx of this variable in order to
        //  follow the one of the "alias" variable
        //  contained in the tape
        idx = pushedVar.idx;
    }
};


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

template <typename T>
static Var<T> const & create_var_and_push(
    T const & value,
    size_t lhs_idx,
    size_t rhs_idx,
    Operation const & op
) {
    auto & tape = Tape<T>::getTape();

    Var<T> const & var =
        tape.push_var(Var<T>{
            value,
            lhs_idx,
            rhs_idx,
            op
        });

    return var;
}

/***************************************************************/
/* SUM */
/***************************************************************/
template <typename T>
Var<T> operator+(Var<T> const & lhs, Var<T> const & rhs) {
    return create_var_and_push(
        lhs.getValue() + rhs.getValue(),
        lhs.getIdx(),
        rhs.getIdx(),
        Operation::SUM
    );
}

template <typename T>
Var<T> operator+(Var<T> const & lhs, T const & rhs) {
    Var<T> const & rhs_var = create_var_and_push(
        rhs,
        0,
        0,
        Operation::NOP
    );

    return create_var_and_push(
        lhs.getValue() + rhs_var.getValue(),
        lhs.getIdx(),
        rhs_var.getIdx(),
        Operation::SUM
    );
}

template <typename T>
Var<T> operator+(T const & lhs, Var<T> const & rhs) {
    return operator+(rhs, lhs);
}



/*------------------------------------------------------------*/
/* forward accumulation                                       */
/*------------------------------------------------------------*/


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
/* SUB                                                         */
/***************************************************************/
template <typename T>
Var<T> operator-(Var<T> const & lhs, Var<T> const & rhs) {
    auto & tape = Tape<T>::getTape();

    Var<T> const & var =
        tape.push_var(Var<T>{
            lhs.getValue() - rhs.getValue(),
            lhs.getIdx(),
            rhs.getIdx(),
            Operation::SUB
        });

    return var;
}


/*------------------------------------------------------------*/
/* forward accumulation                                       */
/*------------------------------------------------------------*/


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
/* MUL                                                         */
/***************************************************************/
template <typename T>
Var<T> operator*(Var<T> const & lhs, Var<T> const & rhs) {
    auto & tape = Tape<T>::getTape();

    Var<T> const & var =
        tape.push_var(Var<T>{
            lhs.getValue() * rhs.getValue(),
            lhs.getIdx(),
            rhs.getIdx(),
            Operation::MUL
        });

    return var;
}

/*------------------------------------------------------------*/
/* forward accumulation                                       */
/*------------------------------------------------------------*/


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
/* DIV                                                         */
/***************************************************************/
template <typename T>
Var<T> operator/(Var<T> const & lhs, Var<T> const & rhs) {
    auto & tape = Tape<T>::getTape();

    Var<T> const & var =
        tape.push_var(Var<T>{
            lhs.getValue() / rhs.getValue(),
            lhs.getIdx(),
            rhs.getIdx(),
            Operation::DIV
        });

    return var;
}

/*------------------------------------------------------------*/
/* forward accumulation                                       */
/*------------------------------------------------------------*/


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


}; // namespace autodiff

#endif // __AUTODIFF__H__
