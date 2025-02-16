#include <iostream>
#include <vector>
#include <memory>

// #include "NodeManager.hpp"
#include "Tape.hpp"

using namespace autodiff::reverse;
using VarD = Var<double>;
using TapeD = Tape<double>;

int main() {

    // NodeManager<double> nm;

    // nm.new_node(1.0);
    // nm.new_node(2.0);
    // nm.new_node(3.0);

    // std::cout << nm.size() << std::endl;
    
    // nm.clear();

    // std::cout << nm.size() << std::endl;
    
    TapeD tape;

    VarD x = tape.var(1.0);
    VarD y = tape.var(2.0);

    // VarD out = x+y;
    
    return 0;
}