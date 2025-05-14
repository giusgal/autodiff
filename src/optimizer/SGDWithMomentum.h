//
// Created by sheldon on 5/14/25.
//

#ifndef SGDWITHMOMENTUM_H
#define SGDWITHMOMENTUM_H
#include <vector>

#include "Optimizer.h"

class SGDWithMomentum : public Optimizer
{
    const double lr;
    const double beta;
    std::vector<double> velocity;
public:
    SGDWithMomentum(double lr, double beta, size_t param_size)
        : lr(lr), beta(beta), velocity(param_size, 0.0){}
    void update(std::vector<double>& params, const std::vector<double>& grads) override
    {
        for (size_t i = 0; i < params.size(); i++)
        {
            velocity[i] = beta * velocity[i] + lr * grads[i];
            params[i] -= velocity[i];
        }
    }
};

#endif //SGDWITHMOMENTUM_H
