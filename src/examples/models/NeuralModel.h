//
// Created on 5/16/25.
//

#ifndef NEURALMODEL_H
#define NEURALMODEL_H

#include "../../autodiff/forward/autodiff.hpp"
#include <vector>
#include <random>
#include <algorithm>

#include "IModel.h"
#include "../optimizer/Optimizer.h"

using namespace autodiff::forward;

struct MLPParams
{
    int H;
    //view splices of a unpacked flat vector

    //2 layer neural network
    static void unpack(const std::vector<DualVar<double>>& p,
        int H,
        std::vector<std::vector<DualVar<double>>>& W1,
        std::vector<DualVar<double>>& b1,
        std::vector<std::vector<DualVar<double>>>& W2,
        DualVar<double>& b2
        )
    {
        // Flat layout: [ W1 (H*D) | b1 (H) | W2 (O*H) | b2 (O=1) ]
        // Here D=1 (single input) and O=1 (single output).
        int idx = 0;
        // W1 is hiddenxinput   because later hidden[i] = W1[i][j] * input[j]
        W1.assign(H, std::vector<DualVar<double>>(1));
        for (int i = 0; i < H; i++)
            W1[i][0] = p[idx++];
        b1.assign(H, DualVar<double>(0,0));
        for (int i = 0; i < H; i++)
            b1[i] = p[idx++];
        //while H2 is outputxhidde,  because out[k] = W2[k][i] * hidden[i]
        W2.assign(1, std::vector<DualVar<double>>(H));
        for (int j = 0; j < H; j++)
            W2[0][j] = p[idx++];
        b2 = p[idx++];

    }
};

class NeuralModel : public IModel
{
    
public:
    void fit(std::vector<std::pair<double, double>>& data) override;

    double predict(double x) const override;

    std::vector<double> get_params() const override;

    void print_parameters() const override;
};


#endif //NEURALMODEL_H