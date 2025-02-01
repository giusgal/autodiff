#ifndef __NODE__HPP__
#define __NODE__HPP__

#include <functional>
#include <string>

namespace autodiff {

template <typename T> class Node;

enum Op {
    NOP,
    SUM,
    SUB,
    MUL,
    DIV
};

inline std::string get_str_from_op(Op const & op) {
    switch(op) {
        case Op::SUM:
            return "+";
            break;
        case Op::SUB:
            return "-";
            break;
        case Op::MUL:
            return "*";
            break;
        case Op::DIV:
            return "/";
            break;
        default:
            return "NOP";
            break;
    }
}

template <typename T>
struct Node {
    // Node(T const & _v): value{_v} {}
    Node() = default;
    Node(
        T const & _v,
        size_t _i,
        size_t _lc,
        size_t _rc,
        Op const & _o
    ):
        value{_v},
        idx{_i},
        left_child{_lc},
        right_child{_rc},
        op{_o},
        grad{0},
        grad_fn{} {}

    bool is_leaf() const {
        return (left_child == 0) && (right_child == 0);
    }

    T value = 0;
    size_t idx = 0;
    size_t left_child = 0;
    size_t right_child = 0;
    Op op = Op::NOP;

    /* gradient */
    T grad = 0;
    std::function<void()> grad_fn{};
};


}; // autodiff

#endif // __NODE__HPP__
