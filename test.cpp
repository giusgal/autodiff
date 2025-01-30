#include <iostream>
#include "autodiff.hpp"



int main() {
    autodiff::DualVar a(4.0, 6.0);
    autodiff::DualVar b(6.0, 2.0);

    std::cout << (autodiff::log(a)).getValue() << std::endl;
    std::cout << (autodiff::pow(a, b)).getValue() << std::endl;
    std::cout << (autodiff::pow(a, 3.0)).getValue() << std::endl;
    std::cout << (autodiff::pow(3.0, a)).getValue() << std::endl;

    std::cout << autodiff::sin(b).getValue() << std::endl;
    std::cout << autodiff::cos(b).getValue() << std::endl;
    std::cout << autodiff::tan(b).getValue() << std::endl;

    return 0;
}
