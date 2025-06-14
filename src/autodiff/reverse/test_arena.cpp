#include <cstddef>
#include <iostream>
#include "ArenaAllocator.hpp"

using namespace autodiff::reverse;

ArenaAllocator arena;

template <typename T>
T * allocate() {
    void * ptr = arena.alloc(sizeof(T), alignof(std::max_align_t));
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
