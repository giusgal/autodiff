//
// Created by sheldon on 6/24/25.
//

#ifndef NEURALMODEL_H
#define NEURALMODEL_H

#include <vector>
#include <utility>
#include <iostream>

#include "IModel.h"
#include "../optimizer/Optimizer.h"
#include "DualVar.hpp"

// Forward declaration from the autodiff library
namespace autodiff { namespace forward {
    template<typename T> class DualVar;
}}

/**
 * @brief A helper struct with static methods to unpack a flat parameter vector
 * into weights (W) and biases (b) for a 2-layer MLP.
 */
struct MLPParams
{
    // Unpacks parameters for use with automatic differentiation (DualVar).
    static void unpack(const std::vector<autodiff::forward::DualVar<double>>& p,
                       int hidden_size,
                       std::vector<std::vector<autodiff::forward::DualVar<double>>>& W1,
                       std::vector<autodiff::forward::DualVar<double>>& b1,
                       std::vector<std::vector<autodiff::forward::DualVar<double>>>& W2,
                       autodiff::forward::DualVar<double>& b2);

    // Unpacks parameters for standard execution (double).
    static void unpack(const std::vector<double>& p,
                       int hidden_size,
                       std::vector<std::vector<double>>& W1,
                       std::vector<double>& b1,
                       std::vector<std::vector<double>>& W2,
                       double& b2);
};

/**
 * @brief A basic implementation of a 2-layer Neural Network (MLP).
 *
 * This class serves as a base model, handling parameter initialization,
 * training loop (fit), and prediction.
 */
class NeuralModel : public IModel
{
protected:
    int epochs, batch_size, hidden_size;
    std::vector<double> params;
    Optimizer* optimizer;

private:
    /**
     * @brief Computes the Mean Squared Error loss for a batch of data.
     * This function is designed to be used with the `gradient` utility.
     */
    autodiff::forward::DualVar<double> loss_func(
        const std::vector<std::pair<double, double>>& batch,
        const std::vector<autodiff::forward::DualVar<double>>& p_dual);

public:
    /**
     * @brief Constructs a new NeuralModel object.
     *
     * @param optimizer The optimization algorithm to use for training.
     * @param hidden_size The number of neurons in the hidden layer.
     * @param epochs The number of passes through the entire dataset.
     * @param batch_size The number of samples per gradient update.
     */
    NeuralModel(Optimizer* optimizer,
                const int hidden_size,
                const int epochs = 50,
                const int batch_size = 10);

    /**
     * @brief Trains the model using the provided data.
     */
    void fit(std::vector<std::pair<double, double>>& data) override;

    /**
     * @brief Makes a prediction for a single input value.
     */
    double predict(double x) const override;

    /**
     * @brief Returns the model's flattened parameter vector.
     */
    std::vector<double> get_params() const override;

    /**
     * @brief Prints the model's parameters to the console.
     */
    void print_parameters() const override;
};


#endif //NEURALMODEL_H
