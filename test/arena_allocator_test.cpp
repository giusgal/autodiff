#include <gtest/gtest.h>
#include "ArenaAllocator.hpp"

/**
 * Unit tests for the functionalities exposed by
 *  ArenaAllocator.hpp
 */

using namespace autodiff::reverse;

constexpr size_t BLOCK_SIZE = 4096;

TEST(ArenaAllocatorTest, StateNewObject) {
    ArenaAllocator<BLOCK_SIZE> arena;
    ASSERT_EQ(arena.n_blocks(), 1);
    ASSERT_EQ(arena.total_size(), 1*BLOCK_SIZE);
    ASSERT_EQ(arena.remaining_size(), BLOCK_SIZE);
    ASSERT_EQ(arena.current_block(), 0);
}

TEST(ArenaAllocatorTest, StateAfterClear) {
    ArenaAllocator<BLOCK_SIZE> arena;

    arena.alloc(BLOCK_SIZE-10, 8);
    arena.alloc(BLOCK_SIZE-10, 8);
    arena.alloc(BLOCK_SIZE-10, 8);
    arena.alloc(BLOCK_SIZE-10, 8);
    arena.alloc(BLOCK_SIZE-10, 8);

    size_t n_block_after_alloc = arena.n_blocks();

    arena.clear();

    ASSERT_EQ(arena.n_blocks(), n_block_after_alloc);
    ASSERT_EQ(arena.current_block(), 0);
    ASSERT_EQ(arena.remaining_size(), BLOCK_SIZE);
}

// This shouldn't compile
// TEST(ArenaAllocatorTest, CannotCopy) {
//     ArenaAllocator<BLOCK_SIZE> arena1;
//     ArenaAllocator<BLOCK_SIZE> arena2;
//     arena2 = arena1;
// }

TEST(ArenaAllocatorTest, Move) {
    ArenaAllocator<BLOCK_SIZE> arena1;
    arena1.alloc(BLOCK_SIZE-10, 8);
    arena1.alloc(BLOCK_SIZE-10, 8);
    arena1.alloc(BLOCK_SIZE-10, 8);
    arena1.alloc(BLOCK_SIZE-10, 8);
    arena1.alloc(BLOCK_SIZE-10, 8);
    size_t n_blocks = arena1.n_blocks();
    size_t current_block = arena1.current_block();
    size_t remaining_size = arena1.remaining_size();
    void * data = arena1.data();

    ArenaAllocator<BLOCK_SIZE> arena2(std::move(arena1));
    ASSERT_EQ(arena2.n_blocks(), n_blocks);
    ASSERT_EQ(arena2.current_block(), current_block);
    ASSERT_EQ(arena2.remaining_size(), remaining_size);
    ASSERT_EQ(arena2.data(), data);

    ArenaAllocator<BLOCK_SIZE> arena3;
    arena3 = std::move(arena2);
    ASSERT_EQ(arena3.n_blocks(), n_blocks);
    ASSERT_EQ(arena3.current_block(), current_block);
    ASSERT_EQ(arena3.remaining_size(), remaining_size);
    ASSERT_EQ(arena3.data(), data);
}

TEST(ArenaAllocator, AllocExceptionNotPowerOfTwo) {
    ArenaAllocator<BLOCK_SIZE> arena;

    EXPECT_THROW(
        arena.alloc(10, 3)
    , std::invalid_argument);
}

TEST(ArenaAllocator, AllocExceptionSizeTooBig) {
    ArenaAllocator<BLOCK_SIZE> arena;

    EXPECT_THROW(
        arena.alloc(BLOCK_SIZE+1, 2)
    , std::bad_alloc);
}
