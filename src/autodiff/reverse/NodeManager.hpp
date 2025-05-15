#ifndef __NODEMANAGER_HPP__
#define __NODEMANAGER_HPP__

#include <vector>
#include <memory>

#include "Node.hpp"
// #include "Functions.hpp"

namespace autodiff {
namespace reverse {

template <typename T>
class NodeManager {
public:
    size_t new_node(T const & value) {
        nodes_.emplace_back(std::make_unique<IndNode<T>>(value));
        return nodes_.size()-1;
    }

    // TODO: (maybe) avoid templates
    // TODO: constraints on NodeType
    template <typename NodeType>
    size_t new_node(size_t first) {
        nodes_.emplace_back(
            std::make_unique<NodeType>(
                nodes_[first].get()
            )
        );
        return nodes_.size()-1;
    }

    // TODO: constraints on NodeType
    template <typename NodeType>
    size_t new_node(size_t first, size_t second) {
        nodes_.emplace_back(
            std::make_unique<NodeType>(
                nodes_[first].get(),
                nodes_[second].get()
            )
        );
        return nodes_.size()-1;
    }

    void clear() {
        nodes_.clear();
    }
    void reserve(size_t dim) {
        nodes_.reserve(dim);
    }

    size_t size() const {
        return nodes_.size();
    }

    void backward(size_t root) {
        // set root node's gradient to default value
        nodes_[root]->update_grad(T{1});

        // nodes are already in topological order
        auto iter = nodes_.rbegin() + (nodes_.size() - root - 1);
        for(; iter != nodes_.rend(); ++iter) {
            (*iter)->backward();
        }
    }

    T get_node_grad(size_t idx) {
        return nodes_[idx]->grad();
    }
    T get_node_value(size_t idx) {
        return nodes_[idx]->value();
    }
private:
    std::vector<std::unique_ptr<Node<T>>> nodes_;
};

}; // namespace reverse
}; // namespace autodiff
 
#endif // __NODEMANAGER_HPP__
