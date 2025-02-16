#ifndef __NODEMANAGER_HPP__
#define __NODEMANAGER_HPP__

#include <vector>
#include <memory>

#include "Node.hpp"
#include "Functions.hpp"

namespace autodiff {
namespace reverse {

template <typename T>
class NodeManager {
public:
    size_t new_node(T const & value) {
        nodes.emplace_back(std::make_unique<IndNode<T>>(value));
        return nodes.size()-1;
    }

    // TODO: constraints on NodeType
    template <typename NodeType>
    size_t new_node(T const & value, size_t first) {
        nodes.emplace_back(std::make_unique<NodeType>(value, first));
        return nodes.size()-1;
    }

    // TODO: constraints on NodeType
    template <typename NodeType>
    size_t new_node(T const & value, size_t first, size_t second) {
        nodes.emplace_back(std::make_unique<NodeType>(value, first, second));
        return nodes.size()-1;
    }

    void clear() {
        nodes.clear();
    }

    size_t size() const {
        return nodes.size();
    }

    void reserve(size_t dim) {
        nodes.reserve(dim);
    }
private:
    std::vector<std::unique_ptr<Node<T>>> nodes;
};

}; // namespace reverse
}; // namespace autodiff
 
#endif // __NODEMANAGER_HPP__
