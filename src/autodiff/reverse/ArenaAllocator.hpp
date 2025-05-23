#ifndef __ARENAALLOCATOR_HPP__
#define __ARENAALLOCATOR_HPP__

#include <cstddef>
#include <cstdint>

namespace autodiff {
namespace reverse {
namespace memory {

class ArenaAllocator {
    using Byte = uint8_t;
public:
    ArenaAllocator(size_t init_capacity = 1);
    ~ArenaAllocator();

    void * Allocate(size_t count);
    void Deallocate();

    size_t Size() const { return size_; }
    size_t Capacity() const { return capacity_; }
private:
    Byte * data_;
    size_t size_;
    size_t capacity_;
};

}; // namespace memory
}; // namespace reverse
}; // namespace autodiff

#endif // __ARENAALLOCATOR_HPP__
