#include "../../../../include/examples/ml/models/NeuralModelOptimized.h"

#include <algorithm> // For std::shuffle
#include <random>    // For std::mt19937

// Constructor implementation
NeuralModelOptimized::NeuralModelOptimized(Optimizer* optimizer,
                                           const int hidden_size,
                                           const int epochs,
                                           const int batch_size)
    : NeuralModel(optimizer, hidden_size, epochs, batch_size) {}

// Private loss function implementation
DualVar<double> NeuralModelOptimized::loss_func_fused(const std::vector<std::pair<double, double>>& batch,
                                                      const std::span<DualVar<double>> p_dual) const
{
    // 1. Unpack parameters from the span
    DualVar<double> total_loss(0, 0);
    auto w1 = p_dual.subspan(0, hidden_size);
    auto b1 = p_dual.subspan(hidden_size, hidden_size);
    auto w2 = p_dual.subspan(2 * hidden_size, hidden_size);
    auto b2 = p_dual.subspan(3 * hidden_size, 1);

    std::vector<DualVar<double>> hidden(hidden_size);
    for (const auto& [x_val, y_val] : batch)
    {
        DualVar<double> x(x_val, 0.0);
        DualVar<double> y(y_val, 0.0);

        // 2. Forward pass: input to hidden layer
        for (int i = 0; i < hidden_size; ++i)
        {
            hidden[i] = tanh(w1[i] * x + b1[i]);
        }

        // 3. Forward pass: hidden to output layer
        DualVar<double> out = b2.back();
        for (int j = 0; j < hidden_size; ++j)
        {
            out = out + hidden[j] * w2[j];
        }

        // 4. Calculate squared error loss for the sample
        DualVar<double> diff = out - y;
        // Using dual number algebra for (diff)^2: (a + bε)^2 = a^2 + 2abε
        total_loss = DualVar<double>(total_loss.getReal() + diff.getReal() * diff.getReal(),
                                   total_loss.getInf() + 2 * diff.getReal() * diff.getInf());
    }

    // 5. Return the average loss for the batch
    return DualVar<double>(total_loss.getReal() / batch.size(),
                           total_loss.getInf() / batch.size());
}

// Public fit method implementation
void NeuralModelOptimized::fit(std::vector<std::pair<double, double>>& data)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        // Randomize data for each epoch to improve training
        std::shuffle(data.begin(), data.end(), std::mt19937(epoch));

        // Process data in mini-batches
        for (int i = 0; i < data.size(); i += batch_size)
        {
            auto batch_end = data.begin() + std::min(i + batch_size, static_cast<int>(data.size()));
            std::vector<std::pair<double, double>> batch(data.begin() + i, batch_end);

            // Compute the gradient for the current batch
            auto grad = gradient<double>(
                [&](const std::vector<DualVar<double>>& p) {
                    return loss_func_fused(batch, {p.data(), p.size()});
                },
                params);

            // Update model parameters using the optimizer
            optimizer->update(params, grad);
        }
    }
}