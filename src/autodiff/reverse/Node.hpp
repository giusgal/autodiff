#ifndef __NODE_HPP__
#define __NODE_HPP__

#include <cstddef>

namespace autodiff {
namespace reverse {

template <typename T> class Node;
template <typename T> class IndNode;
template <typename T> class UnaryNode;
template <typename T> class BinaryNode;

template <typename T>
using NodePtr = Node<T> * const;

// TODO: (maybe) forward() function for lazy evaluation
/**
 * @class Node
 * @brief An abstract class which represents a generic node in the computational graph.
 * @tparam T The type of the underlying variable
 */
template <typename T>
class Node {
public:
    Node(T const & value): value_(value), grad_() {}

    virtual void backward() = 0;
    virtual ~Node() = default;

    T value() const {
        return value_;
    }
    T grad() const {
        return grad_;
    }
    void update_grad(T const & grad) {
        grad_ += grad;
    }
    void clear_grad() {
        grad_ = 0;
    }

protected:
    T value_;
    T grad_;
};

/**
 * @class IndNode
 * @brief A class which represents variables or constants in the computational graph.
 * @tparam T The type of the underlying variable
 *
 * IndNode stands for Independent Node, i.e. the leaf nodes of the computational graph
 */
template <typename T>
class IndNode : public Node<T> {
public:
    IndNode(T const & value): Node<T>(value) {}
    
    // backward on a leaf node does nothing
    void backward() override {}
};


/**
 * @class UnaryNode
 * @brief An abstract class which represents functions taking only one input.
 * @tparam T The type of the underlying variable
 */
template <typename T>
class UnaryNode : public Node<T> {
public:
    UnaryNode(T const & value, NodePtr<T> first):
     Node<T>(value), first_(first) {}

    // virtual void backward() = 0;
protected:
    NodePtr<T> first_;
};

/**
 * @class BinaryNode
 * @brief An abstract class which represents functions taking 2 inputs.
 * @tparam T The type of the underlying variable
 */
template <typename T>
class BinaryNode: public Node<T> {
public:
    BinaryNode(T const & value, NodePtr<T> first, NodePtr<T> second):
     Node<T>(value), first_(first), second_(second) {}

    // virtual void backward() = 0;
protected:
    NodePtr<T> first_;
    NodePtr<T> second_;
};

}; // namespace reverse
}; // namespace autodiff 

#endif // __NODE_HPP__
