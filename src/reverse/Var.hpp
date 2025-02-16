#ifndef __VAR_HPP__
#define __VAR_HPP__

#include "NodeManager.hpp"

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

    T grad(size_t idx) const {
        return manager_ptr_->get_node_grad(idx);
    }

    T value(size_t idx) const {
        return manager_ptr_->get_node_value(idx);
    }

    /******** OPERATORS *******/

private:
    size_t node_idx_;
    NodeManagerPtr<T> manager_ptr_;
};

}; // namespace reverse
}; // namespace autodiff 

#endif // __VAR_HPP__