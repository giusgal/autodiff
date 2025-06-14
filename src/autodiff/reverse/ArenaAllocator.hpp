#ifndef __ARENAALLOCATOR_HPP__
#define __ARENAALLOCATOR_HPP__

#include <cstddef>
#include <memory>
#include <vector>

namespace autodiff {
namespace reverse {

template <size_t BLOCK_SIZE = 4096>
class ArenaAllocator {
public:
    ArenaAllocator() {
        data_ = new std::byte[BLOCK_SIZE];
        blocks_start_.push_back(data_);
        remaining_size_ = BLOCK_SIZE;
        current_block_ = 0;
    }

    ~ArenaAllocator() {
        for(void * block_start: blocks_start_) {
            delete[] static_cast<std::byte*>(block_start);
        }
        data_ = nullptr;
    }

    void * alloc(size_t size, size_t allignment) {
        void * res = std::align(allignment, size, data_, remaining_size_);
        
        if(!res) [[unlikely]] {
            // The aligned object can't be allocated in the current block
            //  => allocate new block or reuse one if it already exists
            // No alignment is necessary in this case because new (malloc)
            //  takes care of this for us
            if(current_block_ + 1 < blocks_start_.size()) {
                // reuse next block
                data_ = blocks_start_[current_block_+1];
            } else {
                // allocate new block
                data_ = new std::byte[BLOCK_SIZE];
                blocks_start_.push_back(data_);
            }
            ++current_block_;
            remaining_size_ = BLOCK_SIZE;
        }


        // TODO: Here we assume that size <= BLOCK_SIZE
        //  We should check wheter this is true or not and in the latter
        //  case we should throw some sort of exception
        void * tmp = data_;
        data_ = static_cast<std::byte*>(data_) + size;
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
private:
    void * data_;
    size_t remaining_size_;
    std::vector<void*> blocks_start_;
    size_t current_block_;
};

}; // namespace reverse
}; // namespace autodiff 

#endif // __ARENAALLOCATOR_HPP__
