//
// Created on 5/16/25.
//

#ifndef NEURALMODEL_H
#define NEURALMODEL_H

#include "../../autodiff/forward/DualVar.hpp"
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
        int hidden_size,
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
        W1.assign(hidden_size, std::vector<DualVar<double>>(1));
        for (int i = 0; i < hidden_size; i++)
            W1[i][0] = p[idx++];
        b1.assign(hidden_size, DualVar<double>(0,0));
        for (int i = 0; i < hidden_size; i++)
            b1[i] = p[idx++];
        //while H2 is outputxhidde,  because out[k] = W2[k][i] * hidden[i]
        W2.assign(1, std::vector<DualVar<double>>(hidden_size));
        for (int j = 0; j < hidden_size; j++)
            W2[0][j] = p[idx++];
        b2 = p[idx++];

    }
    static void unpack(const std::vector<double>& p,
                   int hidden_size,
                   std::vector<std::vector<double>>& W1,
                   std::vector<double>& b1,
                   std::vector<std::vector<double>>& W2,
                   double& b2)
{
    int idx = 0;
    W1.assign(hidden_size, std::vector<double>(1));
    for (int i = 0; i < hidden_size; ++i) W1[i][0] = p[idx++];
    b1.assign(hidden_size, 0.0);
    for (int i = 0; i < hidden_size; ++i) b1[i] = p[idx++];
    W2.assign(1, std::vector<double>(hidden_size));
    for (int j = 0; j < hidden_size; ++j) W2[0][j] = p[idx++];
    b2 = p[idx++];  // now idx == 3*H + 1
}


};

class NeuralModel : public IModel
{
    int epochs, batch_size, hidden_size;
    std::vector<double> params;
    Optimizer* optimizer;

    DualVar<double> loss_func(const std::vector<std::pair<double, double>>& batch,
                              std::vector<DualVar<double>> p_dual
    )
    {
        //1 unpack to my data
        std::vector<std::vector<DualVar<double>>> W1, W2;
        std::vector<DualVar<double>> b1;
        DualVar<double> b2;
        MLPParams::unpack(p_dual, hidden_size, W1, b1, W2, b2);
        DualVar<double> real_accum(0, 0);

        std::vector<DualVar<double>> hidden(hidden_size);
        for (const auto& [x_, y_] : batch)
        {
            //dualvar input to the deep layers
            DualVar<double> x(x_, 0.0);
            DualVar<double> y(y_, 0.0);

            //forward of 1 -> hidden
            for (int i = 0; i < hidden_size; i++)
            {
                //first column is hidden , second is input for w1
                hidden[i] = b1[i] + W1[i][0] * x;
                hidden[i] = relu(hidden[i]);
            }

            //forward of hidden -> 1
            DualVar<double> out = b2;
            for (int j = 0; j < hidden_size; j++)
                //w2 is outxhidden
                out = out + hidden[j] * W2[0][j];

            //now calculate the loss
            DualVar<double> diff = out - y;
            real_accum = DualVar<double>(real_accum.getReal() + diff.getReal()*diff.getReal(),
                real_accum.getInf() + 2 * diff.getReal() * diff.getInf());
            //always using the algebra of dual numbers (a + be) * (c + de)
            // = a*c and a*de + be*c,  be*de = 0..

        }
        //return the accumulated average
        return DualVar<double>(real_accum.getReal() / batch.size(),
            real_accum.getInf() / batch.size());
    }

public:

    NeuralModel(Optimizer* optimizer,
                    const int hidden_size,
                    const int epochs = 50,
                    const int batch_size = 10):
                    epochs(epochs),
    batch_size(batch_size),
    optimizer(optimizer),
    hidden_size(hidden_size)
    {
        // initialize flat params to small random
        std::mt19937 rng(42);
        std::normal_distribution<double> dist(0.0, 0.1);
        int total = 3*hidden_size + 1; // W1 W2 b1 + 1 is b2
        params.resize(total);
        for (auto &v : params)
            v = dist(rng);
    }

    void fit(std::vector<std::pair<double, double>>& data) override
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            //randomize data..
            std::shuffle(data.begin(), data.end(), std::mt19937(epoch));
            //divide the whole dataset into small chuncks
            for (int i = 0; i < data.size(); i+= batch_size)
            {
                auto batch_end = std::min(i + batch_size, static_cast<int>(data.size()));
                std::vector<std::pair<double, double>> batch(data.begin() + i, data.begin() + batch_end);

                //now compute the gradient of these small batch
                auto grad = gradient(
                [&](const std::vector<DualVar<double>>& p){
                    return loss_func(batch, p);
                }, params);

                optimizer->update(params, grad);

            }
        }
    }

    double predict(double x) const override
    {
        // 1) Unpack double params
        std::vector<std::vector<DualVar<double>>> W1, W2;
        std::vector<DualVar<double>> b1;
        DualVar<double> b2;
        std::vector<DualVar<double>> p_dual(params.size());
        for (size_t i = 0; i < params.size(); ++i)
            p_dual[i] = DualVar<double>(params[i], 0.0);

        MLPParams::unpack(p_dual, hidden_size, W1, b1, W2, b2);
        // 2) Forward pass
        // first layer
        std::vector<DualVar<double>> hidden(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            DualVar<double> z = b1[i] + W1[i][0] * x;
            hidden[i] = relu(z);  // ReLU
        }
        // output layer (single output)
        DualVar<double> out = b2;
        for (int j = 0; j < hidden_size; ++j) {
            out = out + hidden[j] * W2[0][j];
        }
        return out.getReal();

    }

    std::vector<double> get_params() const override
    {
        //to be implemented
        return params;  // makes a copy of the flat parameter vector

    }

    void print_parameters() const override
    {
        std::cout << "params = [";
        for (size_t i = 0; i < params.size(); ++i) {
            std::cout << params[i];
            if (i + 1 < params.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }
};


#endif //NEURALMODEL_H