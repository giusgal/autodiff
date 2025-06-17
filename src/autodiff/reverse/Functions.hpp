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

/******* Binary Operators *******/
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

/******* Unary Operators *******/
template <typename T>
class NegNode : public UnaryNode<T> {
public:
    NegNode(NodePtr<T> first):
     UnaryNode<T>(-(first->value()), first) {}

    void backward() override {
        this->first_->update_grad(-(this->grad_));
    }
};

// TODO: maybe this one needs some checks
template <typename T>
class AbsNode : public UnaryNode<T> {
public:
    AbsNode(NodePtr<T> first):
     UnaryNode<T>(std::abs(first->value()), first) {}

    void backward() override {
        int sign = (this->first_->value() >= 0) ? 1 : -1;
        this->first_->update_grad(this->grad_ * sign);
    }
};

template <typename T>
class CosNode : public UnaryNode<T> {
public:
    CosNode(NodePtr<T> first):
     UnaryNode<T>(std::cos(first->value()), first) {}

    void backward() override {
        this->first_->update_grad(this->grad_ * (-std::sin(this->first_->value())));
    }
};

template <typename T>
class SinNode : public UnaryNode<T> {
public:
    SinNode(NodePtr<T> first):
     UnaryNode<T>(std::sin(first->value()), first) {}

    void backward() override {
        this->first_->update_grad(this->grad_ * std::cos(this->first_->value()));
    }
};

template <typename T>
class TanNode : public UnaryNode<T> {
public:
    TanNode(NodePtr<T> first):
     UnaryNode<T>(std::tan(first->value()), first) {}

    void backward() override {
        auto den = std::cos(this->first_->value());
        den *= den;
        this->first_->update_grad(this->grad_ * (1.0/den));
    }
};

template <typename T>
class LogNode : public UnaryNode<T> {
public:
    LogNode(NodePtr<T> first):
     UnaryNode<T>(std::log(first->value()), first) {}

    void backward() override {
        this->first_->update_grad(this->grad_ * (1.0/this->first_->value()));
    }
};

template <typename T>
class ReluNode : public UnaryNode<T> {
public:
    ReluNode(NodePtr<T> first):
     UnaryNode<T>((first->value() > 0.0) ? first->value() : 0.0, first) {}

    void backward() override {
        auto der = (this->first_->value() > 0.0) ? 1.0 : 0.0;
        this->first_->update_grad(this->grad_ * der);
    }
};

template <typename T>
class TanhNode : public UnaryNode<T> {
public:
    TanhNode(NodePtr<T> first):
     UnaryNode<T>(std::tanh(first->value()), first) {}

    void backward() override {
        auto den = std::cosh(this->first_->value());
        den *= den;
        this->first_->update_grad(this->grad_ * (1.0/den));
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

template <typename T>
class SqrtNode : public UnaryNode<T> {
public:
    SqrtNode(NodePtr<T> first):
     UnaryNode<T>(std::sqrt(first->value()), first) {}

    void backward() override {
        T der = 1.0/(2.0*this->value());
        this->first_->update_grad(this->grad_ * der);
    }
};

}; // namespace reverse
}; // namespace autodiff 



#endif // __FUNCTIONS_HPP__
