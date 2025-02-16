#ifndef __FUNCTIONS_HPP__
#define __FUNCTIONS_HPP__

#include "Node.hpp"

namespace autodiff {
namespace reverse {

template <typename T>
class SumNode : public BinaryNode<T> {
public:
    SumNode(T const & value, NodePtr<T> first, NodePtr<T> second):
     BinaryNode<T>(value, first, second) {}

    void backward() override {
        this->first_->update_grad(this->grad_);
        this->second_->update_grad(this->grad_);
    } 
};

template <typename T>
class ProdNode : public BinaryNode<T> {
public:
    ProdNode(T const & value, NodePtr<T> first, NodePtr<T> second):
     BinaryNode<T>(value, first, second) {}

    void backward() override {
        this->first_->update_grad(this->grad_ * this->second_->value_);
        this->second_->update_grad(this->grad_ * this->first_->value_);
    }
};

}; // namespace reverse
}; // namespace autodiff 



#endif // __FUNCTIONS_HPP__