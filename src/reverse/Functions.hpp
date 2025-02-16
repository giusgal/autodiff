#ifndef __FUNCTIONS_HPP__
#define __FUNCTIONS_HPP__

#include "Node.hpp"

namespace autodiff {
namespace reverse {

template <typename T>
class SumNode : public BinaryNode<T> {
public:
    SumNode(T const & value, size_t idx, NodePtr first, NodePtr second):
     BinaryNode<T>(value, idx, first, second) {}

    void backward() override {
        first_->grad_ += this->grad_;
        second_->grad_ += this->grad_;
    } 
};

template <typename T>
class ProdNode : public BinaryNode<T> {
public:
    ProdNode(T const & value, size_t idx, NodePtr first, NodePtr second):
     BinaryNode<T>(value, idx, first, second) {}

    void backward() override {
        first_->grad_ += this->grad_ * second_->value_;
        second_->grad_ += this->grad_ * first_->value_;
    }
};

}; // namespace reverse
}; // namespace autodiff 



#endif // __FUNCTIONS_HPP__