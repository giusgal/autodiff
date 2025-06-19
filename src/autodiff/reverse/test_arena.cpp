#include <charconv>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include "ArenaAllocator.hpp"
#include "Node.hpp"
#include "Functions.hpp"

using namespace autodiff::reverse;

// template <typename T>
// struct is_implicit_lifetime
//     : std::disjunction<
//           std::is_scalar<T>, std::is_array<T>, std::is_aggregate<T>,
//           std::conjunction<
//               std::is_trivially_destructible<T>,
//               std::disjunction<std::is_trivially_default_constructible<T>,
//                                std::is_trivially_copy_constructible<T>,
//                                std::is_trivially_move_constructible<T>>>> {};

ArenaAllocator arena;

template <typename T>
T * allocate() {
    void * ptr = arena.alloc(sizeof(T), alignof(T));
    return std::launder(static_cast<T*>(ptr));
}

template <size_t N>
struct A {
    char a[N];
};

int main() {
    using AddNode = AddNode<double>;
    using SubNode = SubNode<double>;
    using ProdNode = ProdNode<double>;
    using SqrtNode = SqrtNode<double>;
    using ExpNode = ExpNode<double>;

    return 0;
}
