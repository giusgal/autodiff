#ifndef __NEURAL__H__
#define __NEURAL__H__

#include <array>
#include <iostream>
#include <cstddef>
#include <string>
#include <random>
#include <vector>
#include "autodiff.hpp"

#define randomDouble ((double)rand() / (RAND_MAX))
// namespace neural {

using namespace autodiff;
//typedef Var<double> Value;
typedef DualVar<double> Value;

template <typename T>
class Neuron
{
public:
    std::vector<Value> w;
    Value b;

    Neuron(int nin) : b(Value(randomDouble, randomDouble))
    {
        for (int i = 0; i < nin; i++)
        {
            w.emplace_back(Value(randomDouble, randomDouble));
        }
    }

    Value operator()(const std::vector<Value> &x) const
    {
        Value act = b;
        for (size_t i = 0; i < w.size(); i++)
        {
            act = act + (w[i] * x[i]); // not yet defined
            break;
        }
        return autodiff::relu(act); //to be implemented in the Var class
    }

    std::vector<Value> parameters() const
    {
        std::vector<Value> params = w;
        params.push_back(b);
        return params;
    }

private:
    static double randomUniform()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<double> dist(-1.0, 1.0);
        return dist(gen);
    }
};

class Layer
{
public:
    std::vector<Neuron<double>> neurons;

    Layer(int nin, int nout)
    {
        for (int i = 0; i < nout; ++i)
        {
            neurons.emplace_back(nin);
        }
    }

    /*
    std::vector<Value> operator()(const std::vector<Value> &x) const
    {
        std::vector<Value> outs;
        for (const auto &neuron : neurons)
        {
            outs.push_back(neuron(x));
        }
        return outs.size() == 1 ? std::vector<Value>{outs[0]} : outs;
    }
    std::vector<Value> parameters() const
    {
        std::vector<Value> params;
        for (const auto &neuron : neurons)
        {
            auto p = neuron.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
    */

};
    /*
    class MLP
    {
        std::vector<Layer> layers;

    public:
        MLP(const int nin, const std::vector<int> &nouts)
        {
            std::vector<int> sz = {nin};
            sz.insert(sz.end(), nouts.begin(), nouts.end());
            for (size_t i = 0; i < nouts.size(); i++)
            {
                layers.emplace_back(sz[i], sz[i + 1]);
            }
        }

        std::vector<Value> operator()(const std::vector<Value> &x) const
        {
            std::vector<Value> out = x;
            for (const auto &layer : layers)
            {
                out = layer(out);
            }
            return out;
        }

        std::vector<Value> parameters() const
        {
            std::vector<Value> params;
            for (const auto &layer : layers)
            {
                auto p = layer.parameters();
                params.insert(params.end(), p.begin(), p.end());
            }
            return params;
        }
    };
*/

//}; // namespace neural

#endif //__NEURAL__H__
