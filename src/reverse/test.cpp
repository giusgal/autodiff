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
    autodiff::Var z = tape.var(3.0);
    autodiff::Var bb = tape.var(10.0);
    autodiff::Var w = x*x+y*z+bb;

    w.backward();

    std::cout << "∂w/∂x = " << x.grad() << std::endl;
    std::cout << "∂w/∂y = " << y.grad() << std::endl;
    std::cout << "∂w/∂z = " << z.grad() << std::endl;
    std::cout << "z = " << z.val() << std::endl;

    utils::saveGraphToFile(w, "output_graph.png");

    Neuron<double> a(10);
    Layer<double> b(2, 4);
    a.parameters();
    //b.parameters();
    //for(auto& e : b.parameters())
      //  std::cout << e.grad() << std::endl;
    //b.parameters();
    return 0;
}
