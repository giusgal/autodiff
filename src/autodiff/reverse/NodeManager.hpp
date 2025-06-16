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
 * computational graph nodes are automatically created and inserted in a
 * memory pool (arena allocator) in the order of their creation (which naturally forms a valid 
 * topological order of the computational graph).
 * 
 * The class is a singleton in order to force all the allocations to be made in a single
 * memory pool.
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
        static NodeManager instance;
        return instance;
    }

    // *********** Templated factory functions ***********
    // Thanks to templates we can add a new node to the tape
    //  without the need to explicitly define a factory method
    //  for that specific node.
    // The only thing that must be done is to use the appropriate
    //  method based on the general type of node (IndNode/UnaryNode/BinaryNode)
    
    // Factory function for IndNode(s)
    template <typename U>
    friend size_t new_node(U const & value);

    // Factory function for UnaryNode(s)
    template <template <typename> class NodeType, typename U>
    friend size_t new_node(size_t first);

    // Factory function for BinaryNode(s)
    template <template <typename> class NodeType, typename U>
    friend size_t new_node(size_t first, size_t second);

    // *********** Derivatives calculation/update/access ***********
    void backward(size_t root) {
        // set root node's gradient to default value
        nodes_[root]->update_grad(T{1.0});

        // nodes are already in topological order
        auto iter = nodes_.rbegin() + (nodes_.size() - root - 1);
        for(; iter != nodes_.rend(); ++iter) {
            (*iter)->backward();
        }
    }

    void clear_grad() {
        for(auto & node: nodes_) {
            node->clear_grad();
        }
    }

    T get_node_grad(size_t idx) {
        return nodes_[idx]->grad();
    }
    T get_node_value(size_t idx) {
        return nodes_[idx]->value();
    }

    // *********** Utility functions ***********
    void clear() {
        // resets the vector without modifying the capacity
        nodes_.clear();
        
        // resets the arena allocator without releasing the used memory
        arena_.free();
    }

    // TODO: return memory from both the arena and the vector
    // void release() {
    // }

    void reserve(size_t n_nodes) {
        nodes_.reserve(n_nodes);
    }

    size_t size() const {
        return nodes_.size();
    }

private:
    NodeManager() = default;

    // TODO: manage exceptions
    ArenaAllocator<4096> arena_;
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