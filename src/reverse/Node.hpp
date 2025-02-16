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
protected:
    T value_;
    T grad_;
};

template <typename T>
class IndNode : public Node<T> {
public:
    IndNode(T const & value): Node<T>(value) {}
    
    // backward on a leaf node does nothing
    void backward() override {}
};


template <typename T>
class UnaryNode : public Node<T> {
public:
    UnaryNode(T const & value, NodePtr<T> first):
     Node<T>(value), first_(first) {}

    // virtual void backward() = 0;
protected:
    NodePtr<T> first_;
};

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