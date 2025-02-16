#ifndef __NODE_HPP__
#define __NODE_HPP__

namespace autodiff {
namespace reverse {

template <typename T> class Node;
template <typename T> class IndNode;
template <typename T> class UnaryNode;
template <typename T> class BinaryNode;

template <typename T>
using NodePtr = Node<T> * const;

// TODO: (maybe) forward() function for lazy evaluation
template <typename T>
class Node {
public:
    Node(T const & value, size_t idx): value_(value), grad_(), idx_(idx) {}

    virtual void backward() = 0;
    virtual ~Node() = default;
protected:
    T value_;
    T grad_;
    size_t idx_;
};

template <typename T>
class IndNode : public Node<T> {
public:
    IndNode(T const & value, size_t idx): Node<T>(value, idx) {}
    
    // backward on a leaf node does nothing
    void backward() override {}
};


template <typename T>
class UnaryNode : public Node<T> {
public:
    UnaryNode(T const & value, size_t idx, NodePtr first):
     Node<T>(value, idx), first_(first) {}

    virtual void backward() = 0;
protected:
    NodePtr first_;
};

template <typename T>
class BinaryNode: public Node<T> {
public:
    BinaryNode(T const & value, size_t idx, NodePtr first, NodePtr second):
     Node<T>(value, idx), first_(first), second_(second) {}

    virtual void backward() = 0;
protected:
    NodePtr first_;
    NodePtr second_;
};

}; // namespace reverse
}; // namespace autodiff 

#endif // __NODE_HPP__