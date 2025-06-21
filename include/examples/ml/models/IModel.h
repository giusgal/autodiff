//
// Created on 5/16/25.
//

#pragma once

#include <vector>
#include <utility>

class IModel
{
    public:
    virtual ~IModel() = default;

    //train the model on a pair of data x, y
    virtual void fit(std::vector<std::pair<double, double>>& data) = 0;

    //given a single inputx, outputs the prediction y
    virtual double predict (double x) const = 0;

    //access to the model's parameters
    virtual std::vector<double> get_params() const = 0;

    //print the model's parameters
    virtual void print_parameters() const = 0;
};
