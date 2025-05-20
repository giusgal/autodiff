#include <iostream>
#include <vector>
#include <utility>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm> // Include for std::shuffle

// Include headers for your models and optimizers
// Adjust paths as necessary for your project structure
#include <NeuralModel.h>
#include "NeuralModelOptimized.h"
#include "NeuralModelOpenmp.h"
#include "Adam.h"
#include "SGD.h" // Include if you want to test with SGD
#include "IModel.h" // Assuming IModel is the base class

// Define the target function (used for data generation)
#define TARGET_FUNCTION(x) (std::sin(2 * 3.14159265358979323846 * x) + 0.3 * x * x * x)

// A simple helper to generate data with noise
static std::vector<std::pair<double,double>> make_data(int N, double x_min, double x_max) {
    std::vector<std::pair<double,double>> data;
    std::mt19937 rng(42); // Mersenne Twister random number generator with seed
    std::normal_distribution<double> noise(0.0, 0.1); // Noise with mean 0.0 and stddev 0.1

    for(int i = 0; i < N; ++i){
        double x = x_min + (static_cast<double>(i) / (N - 1)) * (x_max - x_min); // x in [x_min, x_max]
        double y = TARGET_FUNCTION(x) + noise(rng); // Add noise to the target
        data.emplace_back(x,y);
    }
    return data;
}

int main(){
    std::cout << "--- Neural Model Hidden Size Scalability Test ---\n";

    // --- Configuration ---
    const int DATA_SIZE = 100; // Fixed data size
    const double X_MIN = -2.0;
    const double X_MAX = 3.0;
    const int EPOCHS = 500; // Number of training epochs
    const int BATCH_SIZE = 5; // Batch size

    // Define the hidden layer sizes to test
    std::vector<int> hidden_sizes = {2, 4, 8, 16, 32, 64, 128}; // Testing a wider range of hidden sizes

    // Generate the data once
    auto data = make_data(DATA_SIZE, X_MIN, X_MAX);
    std::cout << "Generated " << DATA_SIZE << " data points." << std::endl;

    // Output table header
    std::cout << std::left << std::setw(15) << "Hidden Size"
              << std::setw(25) << "Standard Model (ms)"
              << std::setw(25) << "Optimized Model (ms)"
              << std::setw(25) << "OpenMP Model (ms)"
              << "\n";
    std::cout << std::string(90, '-') << "\n"; // Separator line

    // Loop through different hidden sizes
    for (int hidden_size : hidden_sizes) {
        std::cout << std::left << std::setw(15) << hidden_size; // Print current hidden size

        // --- Test Standard Neural Model ---
        // Create a new optimizer for each model instance
        Adam adam_opt_standard(0.03);
        NeuralModel standard_model(&adam_opt_standard, hidden_size, EPOCHS, BATCH_SIZE);

        // Train and time
        auto t0_standard = std::chrono::high_resolution_clock::now();
        standard_model.fit(data);
        auto t1_standard = std::chrono::high_resolution_clock::now();
        auto ms_standard = std::chrono::duration_cast<std::chrono::milliseconds>(t1_standard - t0_standard).count();
        std::cout << std::left << std::setw(25) << ms_standard;

        // --- Test Optimized Neural Model ---
        Adam adam_opt_optimized(0.03);
        NeuralModelOptimized optimized_model(&adam_opt_optimized, hidden_size, EPOCHS, BATCH_SIZE);

        // Train and time
        auto t0_optimized = std::chrono::high_resolution_clock::now();
        optimized_model.fit(data);
        auto t1_optimized = std::chrono::high_resolution_clock::now();
        auto ms_optimized = std::chrono::duration_cast<std::chrono::milliseconds>(t1_optimized - t0_optimized).count();
        std::cout << std::left << std::setw(25) << ms_optimized;

        // --- Test OpenMP Neural Model ---
        Adam adam_opt_openmp(0.03);
        NeuralModelOpenmp openmp_model(&adam_opt_openmp, hidden_size, EPOCHS, BATCH_SIZE);

        // Train and time
        auto t0_openmp = std::chrono::high_resolution_clock::now();
        openmp_model.fit(data);
        auto t1_openmp = std::chrono::high_resolution_clock::now();
        auto ms_openmp = std::chrono::duration_cast<std::chrono::milliseconds>(t1_openmp - t0_openmp).count();
        std::cout << std::left << std::setw(25) << ms_openmp << "\n";
    }

    std::cout << std::string(90, '-') << "\n"; // Separator line
    std::cout << "--- Test Complete ---\n";

    return 0;
}
