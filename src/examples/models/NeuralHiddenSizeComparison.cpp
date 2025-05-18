#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <utility>
#include <string>
#include <sstream> // Required for std::stringstream
#include <vector>  // Ensure std::vector is included

// User's provided header files
#include "LinearModel.h"    // Assuming in the same directory or correct include path
#include "NeuralModel.h"    // Assuming in the same directory or correct include path
#include "Adam.h" // Adjust path as necessary for your project structure

int main() {
    // --- Define the new range for x ---
    const double X_MIN = -2.0;
    const double X_MAX = 3.0;
    const int N = 200; // Number of data points

    // --- Generate Data ---
    std::vector<std::pair<double, double>> data;
    std::mt19937 rng(42); // Mersenne Twister random number generator with seed 42
    std::normal_distribution<double> noise(0.0, 0.1); // Noise with mean 0.0 and stddev 0.1

    std::cout << "Generating " << N << " data points in the range [" << X_MIN << ", " << X_MAX << "]..." << std::endl;
    for (int i = 0; i < N; i++) {
        // Generate x values in the new defined range [X_MIN, X_MAX]
        double x = X_MIN + (static_cast<double>(i) / (N - 1)) * (X_MAX - X_MIN);

        // Target function: sin(2*pi*x) + 0.3*x^3 + noise
        double y =
            std::sin(2 * 3.14159265358979323846 * x) + 0.3 * x * x * x + noise(rng);
        data.emplace_back(x, y);
    }
    std::cout << "Data generation complete." << std::endl;

    // --- Train Linear Model (for reference, as in original logic) ---
    Adam adam_lin_optimizer(0.03);
    LinearModel lin_model(&adam_lin_optimizer, 1000, 16);
    std::cout << "Training Linear Model..." << std::endl;
    lin_model.fit(data);
    std::cout << "Linear Model training complete." << std::endl;

    // --- Define Custom Hidden Layer Sizes for Neural Models ---
    std::vector<int> hidden_layer_sizes = {1, 2, 4, 8, 16};

    std::vector<NeuralModel> neural_models;
    std::vector<Adam> adam_optimizers_nn; // Store optimizers to ensure their lifetime

    // Reserve space for efficiency
    adam_optimizers_nn.reserve(hidden_layer_sizes.size());
    neural_models.reserve(hidden_layer_sizes.size());

    std::cout << "Training Neural Models with specified hidden sizes..." << std::endl;
    for (int h_size : hidden_layer_sizes) {
        adam_optimizers_nn.emplace_back(0.03); // Create a new Adam optimizer for each model
        // Pass optimizer, current h_size, epochs, batch_size
        neural_models.emplace_back(&adam_optimizers_nn.back(), h_size, 1000, 16);

        std::cout << "Training Neural Model with hidden size: " << h_size << std::endl;
        neural_models.back().fit(data); // Train the most recently added model
        std::cout << "Neural Model (h=" << h_size << ") training complete." << std::endl;
    }

    // --- Output Results to CSV ---
    std::ofstream out_file("model_results_hidden_sweep.csv");
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open model_results_hidden_sweep.csv for writing!" << std::endl;
        return 1; // Indicate an error
    }
    std::cout << "Writing results to model_results_hidden_sweep.csv..." << std::endl;

    // Write header row
    out_file << "x,y_true,y_linear";
    for (int h_size : hidden_layer_sizes) {
        out_file << ",y_neural_h" << h_size; // e.g., y_neural_h1, y_neural_h4
    }
    out_file << "\n";

    // Sort data by x for cleaner plotting
    std::sort(data.begin(), data.end(), [](const auto &a, const auto &b) {
        return a.first < b.first;
    });

    // Write data points and model predictions
    for (const auto &pt : data) {
        double x_val = pt.first;
        double y_true_val = pt.second;
        double y_lin_pred = lin_model.predict(x_val);

        out_file << x_val << "," << y_true_val << "," << y_lin_pred;
        // Predictions from each neural model, corresponding to the order in hidden_layer_sizes
        for (size_t i = 0; i < neural_models.size(); ++i) { // Iterate through the trained models
            double y_nn_pred = neural_models[i].predict(x_val);
            out_file << "," << y_nn_pred;
        }
        out_file << "\n";
    }

    out_file.close();
    std::cout << "Results successfully saved to model_results_hidden_sweep.csv" << std::endl;

    return 0; // Indicate successful execution
}
