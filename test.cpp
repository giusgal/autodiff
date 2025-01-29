#include <iostream>
#include "autodiff.hpp"
#include "utils.hpp"

int main() {
    autodiff::Var a{1.0};
    autodiff::Var b{1.0};
    autodiff::Var c{2.0+a};
    autodiff::Var d{c*a};
    autodiff::Var e{d/b};

    std::cout << e.getIdx() << std::endl;

    utils::saveGraphToFile(e, "output_graph.png");

    return 0;
}
