#ifndef __FUNCTIONS_HPP__
#define __FUNCTIONS_HPP__

#include "Node.hpp"

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
class ProdNode : public BinaryNode<T> {
public:
    ProdNode(NodePtr<T> first, NodePtr<T> second):
     BinaryNode<T>(first->value()*second->value(), first, second) {}

    void backward() override {
        this->first_->update_grad(this->grad_ * this->second_->value());
        this->second_->update_grad(this->grad_ * this->first_->value());
    }
};

}; // namespace reverse
}; // namespace autodiff 



#endif // __FUNCTIONS_HPP__