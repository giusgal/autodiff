//
// Created by sheldon on 5/14/25.
//

#pragma once

#include <vector>

#include "Optimizer.h"

/*same sgd but with momentum technique*/
class SGDWithMomentum : public Optimizer
{
    const double lr;
    const double beta;
    std::vector<double> velocity;
public:
    SGDWithMomentum(double lr, double beta, size_t param_size)
        : lr(lr), beta(beta), velocity(param_size, 0.0){}

    void update(std::vector<double>& params, const std::vector<double> grads) override
    {
        for (size_t i = 0; i < params.size(); i++)
        {
            /*the beta and velocity is to gain momentum, a velocity for each parameter to update*/
            velocity[i] = beta * velocity[i] + lr * grads[i];
            params[i] -= velocity[i];
        }
    }
};
