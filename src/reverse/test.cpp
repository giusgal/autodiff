#include <iostream>
#include "autodiff.hpp"
#include "utils.hpp"
#include "neural.hpp"

using namespace std;

int main() {

    autodiff::Tape<double> tape;

    // std::cout << tape.size() << std::endl;

    autodiff::Var x = tape.var(1.0);
    autodiff::Var y = tape.var(2.0);
    autodiff::Var z = tape.var(2.0);
    autodiff::Var w = x/x+y*z;
    autodiff::Var k = w;

    // std::cout << tape.size() << std::endl;

    utils::saveGraphToFile(k, "output_graph.png");

    w.backward();

    Neuron<double> a(10);

    return 0;
}
