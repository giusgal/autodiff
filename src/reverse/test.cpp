#include <iostream>
#include <vector>
#include <memory>

#include "NodeManager.hpp"

using namespace autodiff::reverse;

int main() {

    NodeManager<double> nm;

    nm.new_node(1.0);
    nm.new_node(2.0);
    nm.new_node(3.0);

    std::cout << nm.size() << std::endl;
    
    nm.clear();

    std::cout << nm.size() << std::endl;
    
    return 0;
}