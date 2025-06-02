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

    /******** OPERATORS *******/
    Var<T> operator+(Var<T> const & rhs) const {
        size_t idx = new_node<AddNode, T>(node_idx_, rhs.node_idx_);
        return new_var_from_idx(idx);
    }

    Var<T> operator-(Var<T> const & rhs) const {
        size_t idx = new_node<SubNode, T>(node_idx_, rhs.node_idx_);
        return new_var_from_idx(idx);
    }

    Var<T> operator*(Var<T> const & rhs) const {
        size_t idx = new_node<ProdNode, T>(node_idx_, rhs.node_idx_);
        return new_var_from_idx(idx);
    }

    template <typename U>
    friend Var<U> exp(Var<U> const & arg);

    template <typename U>
    friend Var<U> sqrt(Var<U> const & arg);

private:
    static Var<T> new_var_from_idx(size_t idx) {
        Var<T> tmp;
        tmp.node_idx_ = idx;
        return tmp;
    }

    // The index of the node which is "tracked" by this variable
    size_t node_idx_;
};

/******** OPERATORS *******/
template <typename T>
Var<T> exp(Var<T> const & arg) {
    size_t idx = new_node<ExpNode, T>(arg.node_idx_);
    return Var<T>::new_var_from_idx(idx);
}

template <typename T>
Var<T> sqrt(Var<T> const & arg) {
    size_t idx = new_node<SqrtNode, T>(arg.node_idx_);
    return Var<T>::new_var_from_idx(idx);
}

}; // namespace reverse
}; // namespace autodiff 

#endif // __VAR_HPP__
