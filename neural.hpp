#ifndef __NEURAL__H__
#define __NEURAL__H__

#include <array>
#include <iostream>
#include <cstddef>
#include <string>
#include <random>
#include <vector>
#include "autodiff.hpp"

#define randomDouble ((double) rand() / (RAND_MAX))
namespace neural {

using namespace autodiff;

template <typename T>
class Neuron{
public:
    std::vector<Var<T>> w;
    Var<T> b;

    Neuron(int nin) : b(randomUniform()) {
        for(int i = 0; i < nin; i++){
            w.emplace_back(randomUniform());
        }
    }


    Var<T> operator()(const std::vector<Var<T>>& x)const{
        Var<T> act = b;
        for(size_t i = 0; i < w.size(); i++){
           //act = act + (w(i) * x(i));  not yet defined
           break;
        }
       // return act.tanh(val); to be implemented in the Var class
       return act;
    }

    std::vector<Var<T>> parameters() const {
        std::vector<Var<T>> params = w;
        params.push_back(b);
        return params;
    }

    private:
    static double randomUniform() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<double> dist(-1.0, 1.0);
        return dist(gen);
    }
};

typedef Var<double> Value;

class Layer {
public:
    std::vector<Neuron<double>> neurons;

    Layer(int nin, int nout) {
        for (int i = 0; i < nout; ++i) {
            neurons.emplace_back(nin);
        }
    }

    std::vector<Value> operator()(const std::vector<Value>& x) const {
        std::vector<Value> outs;
        for (const auto& neuron : neurons) {
            outs.push_back(neuron(x));
        }
        return outs.size() == 1 ? std::vector<Value>{outs[0]} : outs;
    }

    std::vector<Value> parameters() const {
        std::vector<Value> params;
        for (const auto& neuron : neurons) {
            auto p = neuron.parameters();
            params.insert(params.end(), p.begin(), p.end());
        }
        return params;
    }
};  


}; // namespace neural

#endif //__NEURAL__H__
