#pragma once

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
    
    /**
     * Factory function for IndNode(s)
     * 
     * @tparam U The type of the underlying variables
     * @param value The value of the `Node`
     */
    template <typename U>
    friend size_t new_node(U const & value);

    /**
     * Factory function for `UnaryNode`(s)
     * 
     * @tparam NodeType The type of the actual `UnaryNode` to create
     * @tparam U The type of the underlying variables
     * @param first The index of the first (and only) argument `Node` of the
     * function the new `Node` represents
     */
    template <template <typename> class NodeType, typename U>
    friend size_t new_node(size_t first);

    /**
     * Factory function for `BinaryNode`(s)
     * 
     * @tparam NodeType The type of the actual `BinaryNode` to create
     * @tparam U The type of the underlying variables
     * @param first The index of the first argument `Node` of the
     * function the new `Node` represents
     * @param second The index of the second argument `Node` of the
     * function the new `Node` represents
     */
    template <template <typename> class NodeType, typename U>
    friend size_t new_node(size_t first, size_t second);

    // *********** Derivatives computation/update/access ***********
    /**
     * Computes the derivative of the `Node` whose index is specified
     * as an argument wrt all the input `Node`(s)
     * 
     * @param root The index of a `Node`
     */
    void backward(size_t root) {
        // Set root node's gradient to default value
        nodes_[root]->update_grad(T{1.0});

        // Nodes are already in topological order
        auto iter = nodes_.rbegin() + (nodes_.size() - root - 1);
        for(; iter != nodes_.rend(); ++iter) {
            (*iter)->backward();
        }
    }

    /**
     * Sets the `grad` field of each `Node` to 0
     */
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
    /**
     * Resets both the vector and the arena allocator without
     * releasing the used memory
     */
    void clear() {
        // resets the vector without modifying the capacity
        nodes_.clear();
        
        // resets the arena allocator without releasing the used memory
        arena_.clear();
    }

    // TODO: return memory from both the arena and the vector
    // void release() {
    // }

    // TODO: delete?
    void reserve(size_t n_nodes) {
        nodes_.reserve(n_nodes);
    }

    /**
     * Returns the number of nodes in the Tape
     */
    size_t size() const {
        return nodes_.size();
    }

private:
    NodeManager() = default;

    // TODO: manage exceptions
    ArenaAllocator<4096> arena_;
    std::vector<Node<T>*> nodes_;
};

// [1] No need for std::launder because we are using
//      the return value of placement new.
//      source: https://youtu.be/5HXCbLilIzs?t=150

template <typename U>
size_t new_node(U const & value) {
    NodeManager<U> & manager = NodeManager<U>::instance();

    void * ptr =
        manager.arena_.alloc(sizeof(IndNode<U>), alignof(IndNode<U>));

    // see [1]
    IndNode<U> * node_ptr = new (ptr) IndNode<U>{value};

    manager.nodes_.emplace_back(node_ptr);

    return manager.nodes_.size()-1;
}

template <template <typename> class NodeType, typename U>
size_t new_node(size_t first) {
    NodeManager<U> & manager = NodeManager<U>::instance();

    void * ptr =
        manager.arena_.alloc(sizeof(NodeType<U>), alignof(NodeType<U>));

    // see [1]
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

    // see [1]
    NodeType<U> * node_ptr = new (ptr) NodeType<U>{
        manager.nodes_[first],
        manager.nodes_[second]
    };

    manager.nodes_.emplace_back(node_ptr);

    return manager.nodes_.size()-1;
}

}; // namespace reverse
}; // namespace autodiff
