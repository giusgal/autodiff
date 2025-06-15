#ifndef __NODEMANAGER_HPP__
#define __NODEMANAGER_HPP__

#include <vector>
#include <memory>
#include <iostream>

#include "Node.hpp"
#include "ArenaAllocator.hpp"

namespace autodiff {
namespace reverse {

/**
 * @class NodeManager
 * @brief A middle-end class between the actual `Node`(s) of the computational graph
 * and the `Var`(s) in the front-end
 * @tparam T The type of the underlying variables
 *
 * The main reason why this class exists is to avoid the need for explicitly computing
 * a topological ordering of the nodes of the computationl graph before the
 * backward pass.
 * In fact, as expressions involving `Var` instances are evaluated, the corresponding
 * computational graph nodes are automatically created and appended to an 
 * `std::vector` in the order of their creation — which naturally forms a valid 
 * topological order of the computational graph.
 */
template <typename T>
class NodeManager {
public:
    // No copy or move allowed
    NodeManager(NodeManager const &) = delete;
    NodeManager& operator=(NodeManager const &) = delete;
    NodeManager(NodeManager &&) = delete;
    NodeManager& operator=(NodeManager &&) = delete;

    static NodeManager& instance() {
        // Guaranteed thread-safe in C++11 and later
        static NodeManager instance_;
        return instance_;
    }

    // Templated factory functions
    template <typename U>
    friend size_t new_node(U const & value);

    // TODO: constraints on NodeType
    template <template <typename> class NodeType, typename U>
    friend size_t new_node(size_t first);

    // TODO: constraints on NodeType
    template <template <typename> class NodeType, typename U>
    friend size_t new_node(size_t first, size_t second);

    void clear() {
        // std::cout << nodes_.size() << std::endl;
        nodes_.clear();
        arena_.free();
    }

    void release() {
        // TODO: return memory from both the arena and the vector
    }

    void reserve(size_t n_nodes) {
        nodes_.reserve(n_nodes);
    }

    void clear_grad() {
        for(auto & node: nodes_) {
            node->clear_grad();
        }
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
    NodeManager() = default;

    // std::vector<std::unique_ptr<Node<T>>> nodes_;

    ArenaAllocator<4096> arena_;
    // TODO: manage exceptions
    std::vector<Node<T>*> nodes_;
};

template <typename U>
size_t new_node(U const & value) {
    NodeManager<U> & manager = NodeManager<U>::instance();

    void * ptr =
        manager.arena_.alloc(sizeof(IndNode<U>), alignof(IndNode<U>));

    IndNode<U> * node_ptr = new (ptr) IndNode<U>{value};

    // TODO: Maybe std::launder?
    manager.nodes_.emplace_back(node_ptr);

    return manager.nodes_.size()-1;
}

template <template <typename> class NodeType, typename U>
size_t new_node(size_t first) {
    NodeManager<U> & manager = NodeManager<U>::instance();

    void * ptr =
        manager.arena_.alloc(sizeof(NodeType<U>), alignof(NodeType<U>));

    NodeType<U> * node_ptr = new (ptr) NodeType<U>{
        manager.nodes_[first]
    };

    manager.nodes_.emplace_back(node_ptr);

    return manager.nodes_.size()-1;
}

template <template <typename> class NodeType, typename U>
size_t new_node(size_t first, size_t second) {
    NodeManager<U> & manager = NodeManager<U>::instance();

    void * ptr =
        manager.arena_.alloc(sizeof(NodeType<U>), alignof(NodeType<U>));

    NodeType<U> * node_ptr = new (ptr) NodeType<U>{
        manager.nodes_[first],
        manager.nodes_[second]
    };

    manager.nodes_.emplace_back(node_ptr);

    return manager.nodes_.size()-1;
}

}; // namespace reverse
}; // namespace autodiff
 
#endif // __NODEMANAGER_HPP__
