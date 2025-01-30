#include <iostream>
#include "autodiff.hpp"
#include "utils.hpp"
#include "neural.hpp"

using namespace std;
int main() {
    autodiff::Var a{1.0};
    autodiff::Var b{1.0};
    autodiff::Var c{2.0+a};
    autodiff::Var d{c*a};
    autodiff::Var e{d/b};
    autodiff::Var f{e + d};
    
    std::cout << f.getIdx() << std::endl;

    neural::Neuron<double> neuron(10);
    for(auto& p: neuron.parameters())
        cout << p.getValue() << endl;

    neural::Layer layer(4, 3);
    layer.parameters();

    utils::saveGraphToFile(f, "output_graph.png");

    return 0;
}
