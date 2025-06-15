#ifndef __ARENAALLOCATOR_HPP__
#define __ARENAALLOCATOR_HPP__

#include <cstddef>
#include <memory>
#include <new>
#include <vector>

namespace autodiff {
namespace reverse {

// Why not std::pmr::unsynchronized_pool_resource?
//  1. Objects of different sizes are put in different pools (not contiguos)
//  2. It's not possible to reuse the same underlying memory multiple times
//  3. Dispatching overhead to decide which pool to use
// Why not std::pmr::monotonic_buffer_resource?
//  1. It's not possible to reuse the same underlying memory multiple times
// => We need a custom allocator

template <size_t BLOCK_SIZE = 4096>
class ArenaAllocator {
    using Byte = std::byte;
public:
    ArenaAllocator():
        remaining_size_{BLOCK_SIZE},
        current_block_{0}
    {
        // TODO: exceptions
        data_ = new Byte[BLOCK_SIZE];
        blocks_start_.push_back(data_);
    }

    ~ArenaAllocator() {
        // TODO: check this
        for(void * block_start: blocks_start_) {
            delete[] static_cast<Byte*>(block_start);
        }
        data_ = nullptr;
    }

    // alignment must be a power of 2 (if not then UB for std::align)
    void * alloc(size_t size, size_t alignment) {
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
                data_ = blocks_start_[current_block_+1];

            } else {
                // allocate new block
                // TODO: exceptions
                data_ = new Byte[BLOCK_SIZE];
                blocks_start_.push_back(data_);

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
        data_ = static_cast<Byte*>(data_) + size;
        remaining_size_ -= size;
        
        return tmp;
    }

    void free() {
        data_ = blocks_start_[0];
        remaining_size_ = BLOCK_SIZE;
        current_block_ = 0;
    }

    size_t n_blocks() const {
        return blocks_start_.size();
    }

    size_t current_block() const {
        return current_block_;
    }

    size_t remaining_size() const {
        return remaining_size_;
    }

    size_t total_size() const {
        return n_blocks()*BLOCK_SIZE;
    }
private:
    void * data_;
    size_t remaining_size_;
    std::vector<void*> blocks_start_;
    size_t current_block_;
};

}; // namespace reverse
}; // namespace autodiff 

#endif // __ARENAALLOCATOR_HPP__
