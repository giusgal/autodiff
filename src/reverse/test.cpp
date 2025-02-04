#include <iostream>
#include "autodiff.hpp"
#include "utils.hpp"

int main() {

    autodiff::Tape<double> tape;

    // std::cout << tape.size() << std::endl;

    auto x = tape.var(3.0);
    auto y = tape.var(4.0);
    // auto z = tape.var(3.0);
    auto z = x*y;
    auto s = (x*x+z)*z;

    // std::cout << tape.size() << std::endl;

    // (x*x+z)*x*y => x*x*x*y + x*x*y*y => 3x^2*y + 2x*y^2 = 6 + 8 = 14

    s.backward();

    std::cout << "∂s/∂x = " << x.grad() << std::endl;
    std::cout << "∂s/∂y = " << y.grad() << std::endl;
    std::cout << "∂s/∂z = " << z.grad() << std::endl;

    utils::save_graph_to_file(s, "output_graph.png");

    return 0;
}
