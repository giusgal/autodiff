#include <iostream>
#include "autodiff.hpp"
#include "neural.hpp"

using namespace std;
int main() {

    //Layer layer(4, 3);
    //layer.parameters();


    
    autodiff::DualVar a(4.0, 6.0);
    autodiff::DualVar b(6.0, 2.0);
    autodiff::DualVar c(-2.0, -1.0);
    autodiff::DualVar d(2.0, -1.0);
    autodiff::DualVar e(-2.0, 1.0);
    autodiff::DualVar f(4.0, 6.0);

    std::cout << (autodiff::log(a)).getValue() << std::endl;
    std::cout << (autodiff::pow(a, b)).getValue() << std::endl;
    std::cout << (autodiff::pow(a, 3.0)).getValue() << std::endl;
    std::cout << (autodiff::pow(3.0, a)).getValue() << std::endl;

    std::cout << autodiff::sin(b).getValue() << std::endl;
    std::cout << autodiff::cos(b).getValue() << std::endl;
    std::cout << autodiff::tan(b).getValue() << std::endl;

    std::cout << "ABS VALUE" << std::endl;
    std::cout << autodiff::abs(c).getValue() << std::endl;
    std::cout << autodiff::abs(d).getValue() << std::endl;
    std::cout << autodiff::abs(d).getValue() << std::endl;

    std::cout << "RELU, ==" << std::endl;
    std::cout << autodiff::relu(c).getValue() << std::endl;
    std::cout << autodiff::relu(d).getValue() << std::endl;
    std::cout << autodiff::relu(e).getValue() << std::endl;
    std::cout << (b == a) << std::endl;
    std::cout << (a == f) << std::endl;

    
    return 0;
}
