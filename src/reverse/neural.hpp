#ifndef __NEURAL__H__
#define __NEURAL__H__

#include <array>
#include <iostream>
#include <cstddef>
#include <string>
#include <random>
#include <vector>
#include "autodiff.hpp"
#include "utils.hpp"


#define randomDouble ((double)rand() / (RAND_MAX))
// namespace neural {

using namespace autodiff;
using Value = autodiff::Var<double>;

Tape<double> tape;

template <typename T>
class Neuron
{
public:
    std::vector<Value> w;
    Value b;

    Neuron(int nin) : b(tape.var(randomDouble))
    {
        for (int i = 0; i < nin; i++)
        {
            w.emplace_back(tape.var(randomDouble));
        }
    }

    // Move constructor (fixes the issue)
    Neuron(Neuron&& other) noexcept
        : w(std::move(other.w)), b(std::move(other.b)) {}

    // Copy constructor (optional, for debugging)
    Neuron(const Neuron& other) = default;

    Value operator()(const std::vector<Value> &x) const
    {
        Value act = b;
        for (size_t i = 0; i < w.size(); i++)
        {
            act = act + (w[i] * x[i]);
        }
        //return autodiff::relu(act); //to be implemented in the Var class
        return act;
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

template <typename T>
class Layer
{
public:
    std::vector<Neuron<T>> neurons;

    Layer(int nin, int nout)
    {
        for (int i = 0; i < nout; ++i)
        {
            neurons.push_back(Neuron<T>(nin));
        }
    }
    std::vector<Value> operator()(const std::vector<Value> &x) const
    {
        std::vector<Value> outs;
        for (const auto &neuron : neurons)
        {
            outs.emplace_back(neuron(x));
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
