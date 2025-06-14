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
    return static_cast<T*>(ptr);
}

template <size_t N>
struct A {
    char a[N];
};

int main() {
    using Type = A<4095>;

    std::cout << sizeof(Type) << std::endl;

    Type * a = allocate<Type>();
    Type * b = allocate<Type>();

    std::cout << arena.n_blocks() << std::endl;
    std::cout << arena.current_block() << std::endl;

    arena.free();

    std::cout << arena.n_blocks() << std::endl;
    std::cout << arena.current_block() << std::endl;

    return 0;
}
