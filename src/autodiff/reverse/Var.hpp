#ifndef __VAR_HPP__
#define __VAR_HPP__

#include "NodeManager.hpp"
#include "Functions.hpp"

namespace autodiff {
namespace reverse {

template <typename T>
using NodeManagerPtr = NodeManager<T>*;

/**
 * @class Var
 * @brief User-facing interface to the `Node`(s) of the computational graph.
 * @tparam T The type of the underlying variable
 */
template <typename T>
class Var {
public:
    Var() = default;

    Var(size_t node_idx, NodeManagerPtr<T> manager_ptr):
     node_idx_(node_idx), manager_ptr_(manager_ptr) {}


    void backward() {
        manager_ptr_->backward(node_idx_);
    }

    T grad() const {
        return manager_ptr_->get_node_grad(node_idx_);
    }

    T value() const {
        return manager_ptr_->get_node_value(node_idx_);
    }

    /******** OPERATORS *******/
    Var<T> operator+(Var<T> const & rhs) const {
        size_t new_node_idx = manager_ptr_-> template new_node<AddNode<T>>(
            node_idx_,
            rhs.node_idx_
        );

        return Var<T>(
            new_node_idx,
            manager_ptr_
        );
    }

    Var<T> operator-(Var<T> const & rhs) const {
        size_t new_node_idx = manager_ptr_-> template new_node<SubNode<T>>(
            node_idx_,
            rhs.node_idx_
        );

        return Var<T>(
            new_node_idx,
            manager_ptr_
        );
    }

    Var<T> operator*(Var<T> const & rhs) const {
        size_t new_node_idx = manager_ptr_-> template new_node<ProdNode<T>>(
            node_idx_,
            rhs.node_idx_
        );

        return Var<T>(
            new_node_idx,
            manager_ptr_
        );
    }

    Var<T> exp() const {
        size_t new_node_idx = manager_ptr_-> template new_node<ExpNode<T>>(
            node_idx_
        );

        return Var<T>(
            new_node_idx,
            manager_ptr_
        );
    }

private:
    // The index of the node which is "tracked" by this variable
    size_t node_idx_;

    // A pointer to the manager object which contains the node
    //  tracked by this variable
    NodeManagerPtr<T> manager_ptr_;
};

}; // namespace reverse
}; // namespace autodiff 

#endif // __VAR_HPP__
