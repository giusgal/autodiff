#ifndef __VAR_HPP__
#define __VAR_HPP__

#include "NodeManager.hpp"
#include "Functions.hpp"

namespace autodiff {
namespace reverse {

/**
 * @class Var
 * @brief User-facing interface to the `Node`(s) of the computational graph.
 * @tparam T The type of the underlying variable
 */
template <typename T>
class Var {
public:
    Var() = default;

    Var(T const & value) {
        node_idx_ = new_node<T>(value);
    }

    void backward() {
        NodeManager<T>::instance().backward(node_idx_);
    }

    T grad() const {
        return NodeManager<T>::instance().get_node_grad(node_idx_);
    }

    T value() const {
        return NodeManager<T>::instance().get_node_value(node_idx_);
    }

    /******** Math functions/operators ********/
    Var<T> operator+() const {
        return *this;
    }

    Var<T> operator+(Var<T> const & rhs) const {
        size_t idx = new_node<AddNode, T>(node_idx_, rhs.node_idx_);
        return new_var_from_idx(idx);
    }

    Var<T> operator+(T const & rhs) const {
        return *this + Var<T>{rhs};
    }

    Var<T> & operator+=(Var<T> const & rhs) {
        node_idx_ = new_node<AddNode, T>(node_idx_, rhs.node_idx_);
        return *this;
    }


    Var<T> operator-() const {
        size_t idx = new_node<NegNode, T>(node_idx_);
        return new_var_from_idx(idx);
    }

    Var<T> operator-(Var<T> const & rhs) const {
        size_t idx = new_node<SubNode, T>(node_idx_, rhs.node_idx_);
        return new_var_from_idx(idx);
    }

    Var<T> operator-(T const & rhs) const {
        return *this - Var<T>{rhs};
    }

    Var<T> & operator-=(Var<T> const & rhs) {
        node_idx_ = new_node<SubNode, T>(node_idx_, rhs.node_idx_);
        return *this;
    }


    Var<T> operator*(Var<T> const & rhs) const {
        size_t idx = new_node<ProdNode, T>(node_idx_, rhs.node_idx_);
        return new_var_from_idx(idx);
    }

    Var<T> operator*(T const & rhs) const {
        return *this * Var<T>{rhs};
    }

    Var<T> & operator*=(Var<T> const & rhs) {
        node_idx_ = new_node<ProdNode, T>(node_idx_, rhs.node_idx_);
        return *this;
    }

    // TODO: implement div

    template <typename U>
    friend Var<U> abs(Var<U> const & arg);

    template <typename U>
    friend Var<U> cos(Var<U> const & arg);

    template <typename U>
    friend Var<U> sin(Var<U> const & arg);

    template <typename U>
    friend Var<U> tan(Var<U> const & arg);

    template <typename U>
    friend Var<U> log(Var<U> const & arg);

    template <typename U>
    friend Var<U> relu(Var<U> const & arg);

    template <typename U>
    friend Var<U> tanh(Var<U> const & arg);

    // TODO: implement pow

    template <typename U>
    friend Var<U> exp(Var<U> const & arg);

    template <typename U>
    friend Var<U> sqrt(Var<U> const & arg);

    /******** Other Operators ********/
    bool operator<(Var<T> const & rhs) const {
        return (value() < rhs.value());
    }
    bool operator<(T const & rhs) const {
        return (value() < rhs);
    }

    bool operator>(Var<T> const & rhs) const {
        return (value() > rhs.value());
    }
    bool operator>(T const & rhs) const {
        return (value() > rhs);
    }

    bool operator==(Var<T> const & rhs) const {
        return (value() == rhs.value());
    }
    bool operator==(T const & rhs) const {
        return (value() == rhs);
    }

    bool operator!=(Var<T> const & rhs) const {
        return !(*this == rhs);
    }
    bool operator!=(T const & rhs) const {
        return !(*this == rhs);
    }

    bool operator<=(Var<T> const & rhs) const {
        return (*this < rhs) || (*this == rhs);
    }
    bool operator<=(T const & rhs) const {
        return (*this < rhs) || (*this == rhs);
    }

    bool operator>=(Var<T> const & rhs) const {
        return (*this > rhs) || (*this == rhs);
    }
    bool operator>=(T const & rhs) const {
        return (*this > rhs) || (*this == rhs);
    }

private:
    static Var<T> new_var_from_idx(size_t idx) {
        Var<T> tmp;
        tmp.node_idx_ = idx;
        return tmp;
    }

    // The index of the node which is "tracked" by this variable
    size_t node_idx_;
};

/******** Math functions/operators *******/
template <typename U>
Var<U> operator+(U const & lhs, Var<U> const & rhs) {
    return Var<U>{lhs} + rhs;
}

template <typename U>
Var<U> operator-(U const & lhs, Var<U> const & rhs) {
    return Var<U>{lhs} - rhs;
}

template <typename U>
Var<U> operator*(U const & lhs, Var<U> const & rhs) {
    return Var<U>{lhs} * rhs;
}

template <typename U>
Var<U> abs(Var<U> const & arg) {
    size_t idx = new_node<AbsNode, U>(arg.node_idx_);
    return Var<U>::new_var_from_idx(idx);
}

template <typename U>
Var<U> cos(Var<U> const & arg) {
    size_t idx = new_node<CosNode, U>(arg.node_idx_);
    return Var<U>::new_var_from_idx(idx);
}

template <typename U>
Var<U> sin(Var<U> const & arg) {
    size_t idx = new_node<SinNode, U>(arg.node_idx_);
    return Var<U>::new_var_from_idx(idx);
}

template <typename U>
Var<U> tan(Var<U> const & arg) {
    size_t idx = new_node<TanNode, U>(arg.node_idx_);
    return Var<U>::new_var_from_idx(idx);
}

template <typename U>
Var<U> log(Var<U> const & arg) {
    size_t idx = new_node<LogNode, U>(arg.node_idx_);
    return Var<U>::new_var_from_idx(idx);
}

template <typename U>
Var<U> relu(Var<U> const & arg) {
    size_t idx = new_node<ReluNode, U>(arg.node_idx_);
    return Var<U>::new_var_from_idx(idx);
}

template <typename U>
Var<U> tanh(Var<U> const & arg) {
    size_t idx = new_node<TanhNode, U>(arg.node_idx_);
    return Var<U>::new_var_from_idx(idx);
}

template <typename U>
Var<U> exp(Var<U> const & arg) {
    size_t idx = new_node<ExpNode, U>(arg.node_idx_);
    return Var<U>::new_var_from_idx(idx);
}

template <typename U>
Var<U> sqrt(Var<U> const & arg) {
    size_t idx = new_node<SqrtNode, U>(arg.node_idx_);
    return Var<U>::new_var_from_idx(idx);
}

}; // namespace reverse
}; // namespace autodiff 

#endif // __VAR_HPP__
