#pragma once

#include <vector>

/*this class can be used to optimize parameters of a model*/
class Optimizer
{
public:
    virtual ~Optimizer() = default;
    virtual void update(std::vector<double> & params, std::vector<double> grads) = 0;
};
