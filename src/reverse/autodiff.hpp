#ifndef __AUTODIFF__HPP__
#define __AUTODIFF__HPP__

#include <cstddef>
#include <set>
#include <vector>

#include "Node.hpp"

namespace autodiff {

template <typename T> class Tape;
template <typename T> class Var;

/*Var**************************************************************************/
template <typename T>
class Var {
public:
    Var(size_t _n, Tape<T> & _t): node_idx{_n}, tape_ref{_t} {}

    size_t get_node_idx() const { return node_idx; }
    Tape<T> & get_tape_ref() const { return tape_ref; }

    void backward();
    T const & grad() const;

    // TODO: delete constructors/operators or do
    //  something different
private:
    /* Helper functions */
    void build_topo(std::vector<size_t> & topo, std::set<size_t> & visited,
            size_t idx);

    size_t node_idx;
    Tape<T> & tape_ref;
};

template <typename T>
void Var<T>::build_topo(std::vector<size_t> & topo, std::set<size_t> & visited, 
        size_t idx) {
    if(visited.find(idx) == visited.end()) {
        visited.emplace(idx);
        
        size_t left_child_idx = tape_ref[idx].left_child;
        size_t right_child_idx = tape_ref[idx].right_child;

        if(left_child_idx != 0) {
            build_topo(topo, visited, left_child_idx);
        }

        if(right_child_idx != 0) {
            build_topo(topo, visited, right_child_idx);
        }

        topo.push_back(idx);
    }
}

template <typename T>
void Var<T>::backward() {
    // topological sort
    std::vector<size_t> topo;
    std::set<size_t> visited;

    build_topo(topo, visited, node_idx);

    // compute gradients
    tape_ref[node_idx].grad = T{1};

    for(auto iter = topo.rbegin(); iter != topo.rend(); ++iter) {
        Node<T> & cur = tape_ref[*iter];
        if(!cur.is_leaf()) {
            cur.grad_fn();
        }
    }

    // clear the tape
    // tape_ref.clear();
}

template <typename T>
T const & Var<T>::grad() const {
    return tape_ref[node_idx].grad;
}

/*Var-Operators*****/
template <typename T>
Var<T> operator+(Var<T> const & lhs, Var<T> const & rhs) {
    Tape<T> & tape = lhs.get_tape_ref();
    size_t lhs_idx = lhs.get_node_idx();
    size_t rhs_idx = rhs.get_node_idx();
    Node<T> & lhs_node = tape[lhs_idx];
    Node<T> & rhs_node = tape[rhs_idx];

    Var<T> new_var = tape.var(
        lhs_node.value + rhs_node.value,
        lhs_node.idx,
        rhs_node.idx,
        Op::SUM
    );
    size_t new_idx = new_var.get_node_idx();

    auto grad_fn = [lhs_idx, rhs_idx, new_idx, &tape]() -> void {
        Node<T> & lhs_node = tape[lhs_idx];
        Node<T> & rhs_node = tape[rhs_idx];
        Node<T> & new_node = tape[new_idx];

        lhs_node.grad += new_node.grad /* *T{1} */;
        rhs_node.grad += new_node.grad /* *T{1} */;
    };
    tape[new_idx].grad_fn = grad_fn;

    return new_var;
}

template <typename T>
Var<T> operator*(Var<T> const & lhs, Var<T> const & rhs) {
    Tape<T> & tape = lhs.get_tape_ref();
    size_t lhs_idx = lhs.get_node_idx();
    size_t rhs_idx = rhs.get_node_idx();
    Node<T> & lhs_node = tape[lhs_idx];
    Node<T> & rhs_node = tape[rhs_idx];

    Var<T> new_var = tape.var(
        lhs_node.value * rhs_node.value,
        lhs_node.idx,
        rhs_node.idx,
        Op::MUL
    );
    size_t new_idx = new_var.get_node_idx();

    auto grad_fn = [lhs_idx, rhs_idx, new_idx, &tape]() -> void {
        Node<T> & lhs_node = tape[lhs_idx];
        Node<T> & rhs_node = tape[rhs_idx];
        Node<T> & new_node = tape[new_idx];

        lhs_node.grad += new_node.grad * rhs_node.value;
        rhs_node.grad += new_node.grad * lhs_node.value;
    };
    tape[new_idx].grad_fn = grad_fn;

    return new_var;
}

template <typename T>
Var<T> operator-(Var<T> const & lhs, Var<T> const & rhs) {
    Tape<T> & tape = lhs.get_tape_ref();
    size_t lhs_idx = lhs.get_node_idx();
    size_t rhs_idx = rhs.get_node_idx();
    Node<T> & lhs_node = tape[lhs_idx];
    Node<T> & rhs_node = tape[rhs_idx];

    Var<T> new_var = tape.var(
        lhs_node.value - rhs_node.value,
        lhs_node.idx,
        rhs_node.idx,
        Op::SUB
    );
    size_t new_idx = new_var.get_node_idx();

    auto grad_fn = [lhs_idx, rhs_idx, new_idx, &tape]() -> void {
        Node<T> & lhs_node = tape[lhs_idx];
        Node<T> & rhs_node = tape[rhs_idx];
        Node<T> & new_node = tape[new_idx];

        lhs_node.grad += new_node.grad;
        rhs_node.grad += (-1)*new_node.grad;
    };
    tape[new_idx].grad_fn = grad_fn;

    return new_var;
}

template <typename T>
Var<T> operator/(Var<T> const & lhs, Var<T> const & rhs) {
    Tape<T> & tape = lhs.get_tape_ref();
    Node<T> const & lhs_node = tape[lhs.get_node_idx()];
    Node<T> const & rhs_node = tape[rhs.get_node_idx()];

    return tape.var(
        lhs_node.value / rhs_node.value,
        lhs_node.idx,
        rhs_node.idx,
        Op::DIV
    );
}
/*Var**************************************************************************/

/*Tape*************************************************************************/
template <typename T>
class Tape {
public:
    Tape();

    void clear();
    size_t size() const;

    Var<T> var(T const &);
    Var<T> var(T const &, size_t, size_t, Op const &);

    Node<T> & operator[](size_t idx) {
        return nodes[idx];
    }

    /*
     * It doesn't make sense for the tape
     *  to be copyed/moved
     * */
    Tape(Tape<T> const &) = delete;
    Tape<T> & operator=(Tape<T> const &) = delete;
    Tape(Tape<T> &&) = delete;
    Tape<T> & operator=(Tape<T> &&) = delete;
private:
    std::vector<Node<T>> nodes;
};

template <typename T>
Tape<T>::Tape() {
    // insert a dummy node as the first node
    nodes.emplace_back();
}

template <typename T>
void Tape<T>::clear() {
    nodes.clear();
    // insert a dummy node as the first node
    nodes.emplace_back();
}

template <typename T>
size_t Tape<T>::size() const {
    return nodes.size();
}

template <typename T>
Var<T> Tape<T>::var(T const & _v) {
    size_t idx = nodes.size();

    nodes.emplace_back(_v,idx,0,0,Op::NOP);

    return Var<T>{
        idx,
        *this
    };
}

template <typename T>
Var<T> Tape<T>::var(T const & _v, size_t _lc, size_t _rc, Op const & _o) {
    size_t idx = nodes.size();

    nodes.emplace_back(_v,idx,_lc,_rc,_o);

    return Var<T>{
        idx,
        *this
    };
}
/*Tape*************************************************************************/


}; // autodiff


#endif // __AUTODIFF__HPP__
