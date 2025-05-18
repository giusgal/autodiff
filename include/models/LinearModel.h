#ifndef LINEARMODEL_H
#define LINEARMODEL_H

#include "../autodiff/forward/DualVar.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include "IModel.h"
#include "../optimizer/Optimizer.h"
using namespace autodiff::forward;

inline std::vector<std::pair<double, double>> generate_data(const int N, const double true_w, const double true_b)
{
    std::default_random_engine rng(42);
    std::normal_distribution<double> noise(0.0, 0.1);
    std::vector<std::pair<double, double>> data;
    for (int i = 0; i < N; i++)
    {
        //generate some random data to learn a mapping
        double x = i * 0.1;
        double y = true_w * x + true_b + noise(rng);
        data.emplace_back(x, y);
    }
    return data;
}


/* this classe is for an example of a simple regression*/
class LinearModel : public IModel{
protected:
    double w;
    double b;
    int epochs;
    int batch_size;
    Optimizer* optimizer;

    DualVar<double> loss_func(const std::vector<std::pair<double, double>>& batch,
                                DualVar<double> w_,
                                DualVar<double> b_)
    {
        double real_accum = 0.0;
        double inf_accum = 0.0;
        for (const auto& [x, y] : batch)
        {
            DualVar x_dual(x, 0.0);
            DualVar y_dual(y, 0.0);
            DualVar<double> y_pred = w_ * x_dual + b_;
            DualVar<double> diff = y_pred - y_dual;
            //this diff * diff  is the loss squared, we want to know dloss/dw and dloss/db
            //the derivative of this would be 2*loss * dloss/dw and 2*loss * dloss/db
            real_accum += (diff * diff).getReal();
            inf_accum += (diff * diff).getInf();
        }
        return DualVar<double>(real_accum / batch.size(), inf_accum / batch.size());
    }

    public:
    virtual ~LinearModel() = default;

    LinearModel(Optimizer* optimizer, int epochs = 50, int batch_size = 10)
        :w(0.0), b(0.0), epochs(epochs), batch_size(batch_size), optimizer(optimizer){}

    virtual void fit(std::vector<std::pair<double, double>>& data) override
    {
        std::vector<double> params = {w, b};
        std::default_random_engine rng(0);
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            std::shuffle(data.begin(), data.end(), rng);
            //this part can be parallel
            for (int i = 0; i < data.size(); i++)
            {
                std::vector<std::pair<double, double>> batch(
                    data.begin() + i,
                    data.begin() + std::min(i + batch_size, static_cast<int>(data.size())));
                auto grad = gradient([&](const std::vector<DualVar<double>> &p)
                {
                    return loss_func(batch, p[0], p[1]);
                }, params);

                optimizer->update(params, grad);
            }

            /*auto loss_val = loss_func(data,
                DualVar<double>(params[0], 0.0),
                DualVar<double>(params[1], 0.0)).getReal();

            std::cout << "Epoch " << epoch
                << " | loss: " << loss_val
                << " | w: " << params[0]
                << " | b: " << params[1] << std::endl;*/
        }
        //std::cout << " | w: " << params[0]
          //      << " | b: " << params[1] << std::endl;
        w = params[0];
        b = params[1];
    }

    void print_parameters()const override
    {
        std::cout << "w: " << w
                << " | b: " << b << std::endl;
    }

    double predict(double x) const override
    {
        return w * x + b;
    }

    std::vector<double> get_params() const override{
        std::vector<double> result(2);
        result[0] = w;
        result[1] = b;
        return result;
    }

};

#endif //LINEARMODEL_H
