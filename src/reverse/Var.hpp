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
    Var(size_t node_idx, NodeManagerPtr manager):
     node_idx_(node_idx), manager_(manager) {}


    void backward() {
        manager_->backward(node_idx_);
    }

    T grad(size_t idx) const {
        return manager_->get_node_grad(idx);
    }

    T value(size_t idx) const {
        return manager_->get_node_value(idx);
    }

private:
    size_t node_idx_;
    NodeManagerPtr manager_;
};

}; // namespace reverse
}; // namespace autodiff 

#endif // __VAR_HPP__