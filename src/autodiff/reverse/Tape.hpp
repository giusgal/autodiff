#ifndef __TAPE_HPP__
#define __TAPE_HPP__

#include "NodeManager.hpp"

namespace autodiff {
namespace reverse {

/**
 * @class Tape
 * @brief User-facing interface to the NodeManager class.
 * @tparam T The type of the underlying variables
 */
// template <typename T>
// class Tape {
// public:
//     Tape() = default;
//     Var<T> var(T const & value) {
//         size_t idx = manager_.new_node(value);
//         return Var<T>(idx, &manager_);
//     }
//
//     // a Tape cannot be copyied nor moved
//     // TODO: maybe it can be moved
//     Tape(Tape const &) = delete;
//     Tape& operator=(Tape const &) = delete;
//     Tape(Tape &&) = delete;
//     Tape& operator=(Tape &&) = delete;
// private:
//     NodeManager<T> manager_;
// };

template <typename T>
class Tape {
public:
    static Tape& instance() {
        // Guaranteed thread-safe in C++11 and later
        static Tape instance_;
        return instance_;
    }

    // Var<T> var(T const & value) {
    //     size_t idx = manager_.new_node(value);
    //     return Var<T>(idx, &manager_);
    // }

    NodeManager<T>& manager() {
        return manager_;
    }

    // No copy or move allowed
    Tape(Tape const &) = delete;
    Tape& operator=(Tape const &) = delete;
    Tape(Tape &&) = delete;
    Tape& operator=(Tape &&) = delete;
private:
    Tape() = default;
    ~Tape() = default;

    NodeManager<T> manager_;
};

}; // namespace autodiff
}; // namespace reverse

#endif // __TAPE_HPP__
