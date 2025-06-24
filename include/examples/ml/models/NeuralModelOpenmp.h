//
// Created by sheldon on 6/24/25.
//

#ifndef NEURALMODELOPENMP_H
#define NEURALMODELOPENMP_H

#include <vector>
#include <utility>
#include <span>

#include "NeuralModel.h" // Provides base class, Optimizer, DualVar, etc.

/**
 * @brief A neural network model that uses OpenMP to parallelize the training process
 * across mini-batches.
 */
class NeuralModelOpenmp2 : public NeuralModel
{
public:
    /**
     * @brief Construct a new NeuralModelOpenmp1 object.
     *
     * @param optimizer Pointer to the optimizer for updating model parameters.
     * @param hidden_size The number of neurons in the hidden layer.
     * @param epochs The number of training epochs.
     * @param batch_size The size of each mini-batch for gradient calculation.
     */
    NeuralModelOpenmp2(Optimizer* optimizer,
                       const int hidden_size,
                       const int epochs = 50,
                       const int batch_size = 10);

    /**
     * @brief Trains the model on the provided data using OpenMP for parallelization.
     *
     * The method divides the dataset into meta-batches, and within each meta-batch,
     * it processes individual mini-batches in parallel using OpenMP threads.
     * The gradients from each thread are then aggregated and averaged before updating
     * the model parameters.
     *
     * @param data The training dataset, a vector of input-output pairs.
     */
    void fit(std::vector<std::pair<double, double>>& data) override;

private:
    /**
     * @brief Computes the loss for a given batch of data. This function is designed to be
     * called by the autodiff `gradient` utility.
     *
     * This implementation also uses an internal OpenMP parallel for loop to speed up
     * the loss calculation over the samples within a single batch.
     *
     * @param batch A mini-batch of training data.
     * @param p_dual The model parameters (weights and biases) as a span of dual variables.
     * @return The calculated loss as a dual variable, containing both the real loss and its derivative part.
     */
    DualVar<double> loss_func_fused(const std::vector<std::pair<double, double>>& batch,
                                    const std::span<const DualVar<double>> p_dual) const;
};

#endif //NEURALMODELOPENMP_H
