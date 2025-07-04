#pragma once

#include "Optimizer.h"

/* the simplest gradient descent */
class SGD : public Optimizer
{
    const double lr;
public:
    SGD(const double lr): lr(lr) {};
    void update(std::vector<double>& params, const std::vector<double> grads) override
    {
        for (size_t i = 0; i < params.size(); i++)
        {
            params[i] -= lr * grads[i];
        }
    }

};
