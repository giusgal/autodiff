#include "ArenaAllocator.hpp"

#include <bit>
#include <algorithm>

using namespace autodiff::reverse::memory;

ArenaAllocator::ArenaAllocator(size_t init_capacity):
    data_{new Byte[init_capacity]},
    size_{0},
    capacity_{init_capacity}
{}

ArenaAllocator::~ArenaAllocator() {
    Deallocate();
}

void * ArenaAllocator::Allocate(size_t count) {
    // relocation check
    if(size_ + count > capacity_) {

        // if size_ + count > 2*capacity_ then:
        //  the new capacity becomes the smallest
        //  integral power of two which is greater than
        //  or equal to size_ + count
        // else:
        //  the new capacity is twice the old capacity
        size_t new_capacity =
            (size_ + count > 2*capacity_)
                ? std::bit_ceil(size_+count)
                : 2*capacity_;

        // allocate new data array
        Byte * new_data = new Byte[new_capacity];

        // copy old data into the new array
        std::copy(data_, data_+size_, new_data);
        
        // deallocate old array
        delete[] data_;
        data_ = new_data;
        capacity_ = new_capacity;
    }

    void * ptr = data_+size_;
    size_ = size_ + count;
    return ptr;
}

void ArenaAllocator::Deallocate() {
    delete[] data_;
    data_ = nullptr;
}
