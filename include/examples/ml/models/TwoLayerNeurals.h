#pragma once

#include <vector>
//#include "DualVar.hpp"
#include <iostream>
#include <random>
#include <algorithm>
#include "Optimizer.h"

class TwoLayerNeurals
{
    std::vector<std::vector<double>> weights1;
    std::vector<std::vector<double>> weights2;
    std::vector<double> bias1;
    std::vector<double> bias2;
    int input_size, hidden_size, output_size;


};
