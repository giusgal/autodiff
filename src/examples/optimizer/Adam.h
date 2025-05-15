//
// Created by sheldon on 5/14/25.
//

#ifndef ADAM_H
#define ADAM_H
#include <bits/valarray_after.h>
#include <bits/valarray_after.h>
#include <bits/valarray_after.h>
#include <cmath>
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
       double beta1 = 0.9,
       double beta2 = 0.999,
       double epsilon = 1e-8)
      : lr(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

  void update(std::vector<double> &params, std::vector<double> grad) override
  {
      if (m.empty()) // this could done outside but in this case it is to adapt the size of param
      {
          m.resize(params.size(), 0.0);
          v.resize(params.size(), 0.0);
      }
      t++;
      for (size_t i = 0; i < params.size(); i++)
      {
          m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
          v[i] = beta2 * v[i] + (1 - beta2) * grad[i];

          // to not make m and v too small (biased toward zero) in early times, later it becomes 1-0 so divided by 1
          double m_hat = m[i] / (1 - std::pow(beta1, t));
          double v_hat = v[i] / (1 - std::pow(beta2, t));

          //m_hat is the moving average of the gradients at time 5
          params[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
      }
  }
};


#endif //ADAM_H
