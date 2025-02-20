#ifndef __TAPE_HPP__
#define __TAPE_HPP__

#include "NodeManager.hpp"
#include "Var.hpp"

namespace autodiff {
namespace reverse {

template <typename T>
class Tape {
public:
    Tape() = default;
    Var<T> var(T const & value) {
        size_t idx = manager_.new_node(value);
        return Var<T>(idx, &manager_);
    }

    // a Tape cannot be copyied nor moved
    // TODO: maybe it can be moved
    Tape(Tape const &) = delete;
    Tape& operator=(Tape const &) = delete;
    Tape(Tape &&) = delete;
    Tape& operator=(Tape &&) = delete;
private:
    NodeManager<T> manager_;
};

}; // namespace autodiff
}; // namespace reverse

#endif // __TAPE_HPP__