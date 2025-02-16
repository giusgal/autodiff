#ifndef __VAR_HPP__
#define __VAR_HPP__

#include "NodeManager.hpp"
#include "Functions.hpp"

namespace autodiff {
namespace reverse {

template <typename T>
using NodeManagerPtr = NodeManager<T>*;

template <typename T>
class Var {
public:
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
    Var<T> operator+(Var<T> const & rhs) {
        #ifdef AUTODIFF_REVERSE_VAR_CHECK_MANAGER
        // TODO
        #endif
        T lhs_value = manager_ptr_->get_node_value(node_idx_);
        T rhs_value = manager_ptr_->get_node_value(rhs.node_idx_);

        size_t new_node_idx = manager_ptr_-> template new_node<AddNode<T>>(
            lhs_value+rhs_value,
            node_idx_,
            rhs.node_idx_
        );

        return Var<T>(
            new_node_idx,
            manager_ptr_
        );
    }

private:
    size_t node_idx_;
    NodeManagerPtr<T> manager_ptr_;
};

}; // namespace reverse
}; // namespace autodiff 

#endif // __VAR_HPP__