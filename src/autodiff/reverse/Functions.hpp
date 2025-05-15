#ifndef __FUNCTIONS_HPP__
#define __FUNCTIONS_HPP__

#include "Node.hpp"
#include <cmath>

/**
 * The actual definition for all the functions that inherit from the
 * `Node` class goes here
 */

namespace autodiff {
namespace reverse {

template <typename T>
class AddNode : public BinaryNode<T> {
public:
    AddNode(NodePtr<T> first, NodePtr<T> second):
     BinaryNode<T>(first->value()+second->value(), first, second) {}

    void backward() override {
        this->first_->update_grad(this->grad_);
        this->second_->update_grad(this->grad_);
    } 
};

template <typename T>
class SubNode : public BinaryNode<T> {
public:
    SubNode(NodePtr<T> first, NodePtr<T> second):
     BinaryNode<T>(first->value()-second->value(), first, second) {}

    void backward() override {
        this->first_->update_grad(this->grad_);
        this->second_->update_grad(-(this->grad_));
    } 
};

template <typename T>
class ProdNode : public BinaryNode<T> {
public:
    ProdNode(NodePtr<T> first, NodePtr<T> second):
     BinaryNode<T>(first->value()*second->value(), first, second) {}

    void backward() override {
        this->first_->update_grad(this->grad_ * this->second_->value());
        this->second_->update_grad(this->grad_ * this->first_->value());
    }
};

template <typename T>
class ExpNode : public UnaryNode<T> {
public:
    ExpNode(NodePtr<T> first):
     UnaryNode<T>(std::exp(first->value()), first) {}

    void backward() override {
        this->first_->update_grad(this->grad_ * this->value());
    }
};

}; // namespace reverse
}; // namespace autodiff 



#endif // __FUNCTIONS_HPP__
