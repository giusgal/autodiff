#ifndef ADAM_H
#define ADAM_H

#include <cmath>
#include <algorithm>
#include <iostream>
#include "Optimizer.h"

//  Adaptive Moment Estimation
class Adam : public Optimizer
{
    std::vector<double> m;
    std::vector<double> v;
    double beta1;
    double beta2;
    double epsilon;
    double lr;
    int t = 0;
    //beta1 controls momentum of gradient
    //beta2 controls adaption of the learning rate
    //eps is a small value to prevent division by zero
    //t is the timestep to be increased

public:
    Adam(double learning_rate = 0.001,
         double beta1        = 0.9,
         double beta2        = 0.999,
         double epsilon      = 1e-8)
      : lr(learning_rate),
        beta1(beta1),
        beta2(beta2),
        epsilon(epsilon)
    {}

    void update(std::vector<double> &params,
                const std::vector<double> grads) override
    {
        if (m.empty()) // this could be done outside but here we adapt size to params
        {
            m.resize(params.size(), 0.0);
            v.resize(params.size(), 0.0);
        }

        t++;
        for (size_t i = 0; i < params.size(); i++)
        {
            // 1) Update biased first & second moment estimates
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
            //this part of gradient square is a solution to converge, otherwise
            //v goes big..

            // 2) Compute bias-correction denominators, clamped to epsilon
            double denom1 = 1.0 - std::pow(beta1, t);
            double denom2 = 1.0 - std::pow(beta2, t);
            denom1 = std::max(denom1, epsilon);
            denom2 = std::max(denom2, epsilon);

            // 3) Compute bias-corrected moments
            double m_hat = m[i] / denom1;
            double v_hat = v[i] / denom2;

            // 4) Parameter update: epsilon inside sqrt for stability
            params[i] -= lr * m_hat / std::sqrt(v_hat + epsilon);
        }
    }
};

#endif // ADAM_H
