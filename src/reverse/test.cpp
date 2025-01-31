#include <iostream>
#include "autodiff.hpp"
#include "utils.hpp"

int main() {

    autodiff::Tape<double> tape;

    std::cout << tape.size() << std::endl;

    autodiff::Var x = tape.var(1.0);
    autodiff::Var y = tape.var(2.0);
    autodiff::Var z = tape.var(2.0);
    autodiff::Var w = x+y*z;

    std::cout << tape.size() << std::endl;

    utils::saveGraphToFile(w, "output_graph.png");

    return 0;
}
