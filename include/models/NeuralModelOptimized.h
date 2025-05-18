//
// Created by sheldon on 5/18/25.
//

#ifndef NEURALMODELOPTIMIZED_H
#define NEURALMODELOPTIMIZED_H
#include <span>

#include "NeuralModel.h"

class NeuralModelOptimized : public NeuralModel
{
    DualVar<double> loss_func_fused(const std::vector<std::pair<double, double>>& batch,
                              const std::span<DualVar<double>> p_dual
    ) const
    {
        //1 unpack to my data
        DualVar<double> real_accum(0, 0);

        auto w1 = p_dual.subspan(0, hidden_size);
        auto b1 = p_dual.subspan(hidden_size, hidden_size);
        auto w2 = p_dual.subspan(2 * hidden_size, hidden_size);
        DualVar<double> b2 = p_dual[3 * hidden_size];

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
                hidden[i] = tanh(w1[i] * x + b1[i]);
            }

            //forward of hidden -> 1
            DualVar<double> out = b2;
            for (int j = 0; j < hidden_size; j++)
                //w2 is outxhidden
                    out = out + hidden[j] * w2[j];

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
    NeuralModelOptimized(Optimizer* optimizer,
                const int hidden_size,
                const int epochs = 50,
                const int batch_size = 10): NeuralModel(optimizer, hidden_size, epochs, batch_size){}

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
                    std::vector<DualVar<double>> p_span(p);
                    return loss_func_fused(batch, p_span);
                }, params);

                optimizer->update(params, grad);

            }
        }
    }

};

#endif //NEURALMODELOPTIMIZED_H
