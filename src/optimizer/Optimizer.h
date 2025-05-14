
#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <vector>

class Optimizer
{
public:
    virtual ~Optimizer() = default;
    virtual void update(std::vector<double> & params, const std::vector<double> grads) = 0;
};

#endif //OPTIMIZER_H
