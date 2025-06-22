#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace autodiff {
namespace reverse {

// Why not std::pmr::unsynchronized_pool_resource?
//  1. Objects of different sizes are put in different pools (not contiguous)
//  2. It's not possible to reuse the same underlying memory multiple times
//  3. Dispatching overhead to decide which pool to use
// Why not std::pmr::monotonic_buffer_resource?
//  1. It's not possible to reuse the same underlying memory multiple times
// => We need a custom allocator

/**
 * @class ArenaAllocator
 * @brief Memory pool that can dynamically increase in size and that allows to reuse
 * the underlying memory multiple times
 * @tparam BLOCK_SIZE The size of the blocks that are allocated
 * 
 * EXAMPLE:
 *     Block#0                 Block#1                    Block#N             
 *     +---+-+-----+------+    +----+-------+-----+       +------------------+
 *     |   |x|     |      |    |    |       |     |       |                  |
 *     |obj|x| obj | obj  |    |obj |  obj  |     |  ...  |                  |
 *     |   |x|     |      |    |    |       |     |       |                  |
 *     +---+-+-----+------+    +----+-------+-----+       +------------------+
 *          ^                               ^             ^
 *          |                               |             |                   
 *          Unused space due to             data_         A block that is not currently
 *          alignment                                     used that was previously allocated
 */
template <size_t BLOCK_SIZE = 4096>
class ArenaAllocator {
    using Byte = std::byte;
public:
    // Copy not allowed
    // Why?
    //  Actually we could copy the content of the arena
    //  into another arena but:
    //   1. There aren't many cases where a copy would
    //      be useful
    //   2. The operation would be slow for arenas that
    //      contain a lot of objects
    ArenaAllocator(ArenaAllocator const &) = delete;
    ArenaAllocator& operator=(ArenaAllocator const &) = delete;
    
    // Move allowed
    //  (we need to declare them because deleting the copy and
    //  copy-assignment, deletes these as well)
    ArenaAllocator(ArenaAllocator &&) = default;
    ArenaAllocator& operator=(ArenaAllocator &&) = default;

    ArenaAllocator():
        remaining_size_{BLOCK_SIZE},
        current_block_{0}
    {
        // Note: by doing this we avoid memory leaks
        auto new_block = std::make_unique<Byte[]>(BLOCK_SIZE);
        blocks_start_.emplace_back(std::move(new_block));
        data_ = blocks_start_.back().get();
    }

    /**
     * Returns a pointer to a region of memory where an object of
     * size `size` and with alignment constraint `alignment`
     * can be constructed
     * 
     * @param size The size of the object to be constructed
     * @param alignment Alignment constraint of the object
     * 
     * alignment must be a power of 2 (if not then UB for std::align)
     */
    void * alloc(size_t const size, size_t const alignment) {
        bool is_power_of_2 = alignment > 0 && !(alignment & (alignment-1));
        if(!is_power_of_2) {
            throw std::invalid_argument("alignment must be a power of 2");
        }

        if(size > BLOCK_SIZE) [[unlikely]] {
            throw std::bad_alloc();
        }

        // Align the "data_" pointer for the next allocation
        // Note that:
        //  i)  On success => This function modifies both the "data_"
        //      pointer and the "remaining_size_" variable.
        //  ii) On failure => "res" is nullptr and no updates to the variables
        //      take place.
        void * res = std::align(alignment, size, data_, remaining_size_);
        
        if(!res) [[unlikely]] {

            // The aligned object can't be allocated in the current block
            //  => allocate new block or reuse one if it already exists

            if(current_block_ + 1 < blocks_start_.size()) {
                // reuse next block
                data_ = blocks_start_[current_block_+1].get();

            } else {
                // allocate new block
                // Note: by doing this we avoid memory leaks
                auto new_block = std::make_unique<Byte[]>(BLOCK_SIZE);
                blocks_start_.emplace_back(std::move(new_block));
                data_ = blocks_start_.back().get();

            }

            ++current_block_;
            remaining_size_ = BLOCK_SIZE;

            // Now we must align the "data_" pointer wrt the "alignment"
            //  parameter
            // This is not strictly necessary because if we are using
            //  a new block then "new" (malloc) has already taken care
            //  of the alignment for us, but here we are considering an
            //  "alignment" parameter so it must be done just to be sure.
            res = std::align(alignment, size, data_, remaining_size_);

            // This can happen if, for some strange allignment constraints,
            //  the allocation request doesn't fit in the empty block even tough
            //  the size of the request is <= BLOCK_SIZE.
            //  (This is very unlikely especially given the use of the arena
            //  allocator in this library but must be checked)
            if(!res) {
                throw std::bad_alloc();
            }
        }

        // Move the "data_" pointer forward for the next allocation and
        //  shrink the "remaining_size_"
        void * tmp = data_;
        data_ = reinterpret_cast<Byte*>(data_) + size;
        remaining_size_ -= size;
        
        return tmp;
    }

    /**
     * This functions deos nothing because this ArenaAllocator
     *  deallocates (i.e. returns the used memory to malloc/OS)
     *  only when it is destroyed
     */
    // void dealloc() {}

    /**
     * Resets the arena allocator without releasing the used memory
     */
    void clear() {
        data_ = blocks_start_[0].get();
        remaining_size_ = BLOCK_SIZE;
        current_block_ = 0;
    }

    size_t n_blocks() const { return blocks_start_.size(); }
    size_t current_block() const { return current_block_; }
    size_t remaining_size() const { return remaining_size_; }
    size_t total_size() const { return n_blocks()*BLOCK_SIZE; }
    void * data() const { return data_; }
private:
    void * data_;
    size_t remaining_size_;
    std::vector<std::unique_ptr<Byte[]>> blocks_start_;
    size_t current_block_;
};

}; // namespace reverse
}; // namespace autodiff 
