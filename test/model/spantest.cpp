#include <iostream>
#include <vector>
#include <utility>
#include <chrono>
#include <random> // Include for std::mt19937 and std::normal_distribution
#include <iomanip> // For output formatting

// Include headers for your models and optimizers
#include <models/NeuralModel.h>
#include <models/NeuralModelOptimized.h> // Include the header for your optimized model
#include <optimizer/Adam.h>
#include <optimizer/SGD.h>
#include <models/IModel.h> // Assuming IModel is the base class

// Define the target function
#define TARGET_FUNCTION(x) (5.0 *(x) - 1.0) // Example: y = 5x - 1
// This definition appears unused in the provided main function,
// as TARGET_FUNCTION is used instead.
// #define TARGET_FUNCTION1(x) (3.0 * (x) * (x) + 2.0 * (x) - 5.0) // Example: y = 3x^2 + 2x - 5


// A simple helper to generate data y = 5x - 1 with noise
static std::vector<std::pair<double,double>> make_data(int N) {
    std::vector<std::pair<double,double>> data;
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, 0.05); // Add some noise
    for(int i = 0; i < N; ++i){
        double x = (i / double(N - 1)) * 2.0 - 1.0;  // x in [−1, +1]
        double y = TARGET_FUNCTION(x) + noise(rng); // Add noise to the target
        data.emplace_back(x,y);
    }
    return data;
}

int main(){
    std::cout << "--- Neural Model Scalability Test ---\n";

    // Define the configurations to test
    std::vector<int> hidden_sizes = {2, 4, 8, 16};
    std::vector<int> data_sizes = {200, 600};

    // Output table header
    std::cout << std::left << std::setw(15) << "Data Size" << std::endl
              << std::setw(15) << "Hidden Size"
              << std::setw(25) << "Standard Model (ms)"
              << std::setw(25) << "Optimized Model (ms)"
              << "\n";
    std::cout << std::string(80, '-') << "\n"; // Separator line

    // Loop through data sizes
    for (int data_size : data_sizes) {
        // Generate data for the current size
        auto data = make_data(data_size);
        std::cout << std::left << std::setw(15) << data_size << std::endl; // Print data size once per row

        // Loop through hidden sizes
        for (int hidden_size : hidden_sizes) {
            // Print hidden size for the current column
            std::cout << std::left << std::setw(15) << hidden_size;

            // --- Standard Neural Model Training ---
            // Create a new optimizer for each run to ensure a clean state
            Adam adam_opt_standard(0.01);
            // Or SGD: SGD sgd_opt_standard(0.01);

            // Instantiate Standard NeuralModel
            NeuralModel standard_model(&adam_opt_standard, hidden_size, /*epochs=*/500, /*batch_size=*/16); // Reduced epochs for faster testing
            // If using SGD: NeuralModel standard_model(&sgd_opt_standard, hidden_size, /*epochs=*/500, /*batch_size=*/16);

            // Train and time the Standard Neural Model
            auto t0_standard = std::chrono::high_resolution_clock::now();
            standard_model.fit(data);
            auto t1_standard = std::chrono::high_resolution_clock::now();
            auto ms_standard = std::chrono::duration_cast<std::chrono::milliseconds>(t1_standard - t0_standard).count();

            // --- Optimized Neural Model Training ---
            // Create a new optimizer for each run
            Adam adam_opt_optimized(0.01);
            // Or SGD: SGD sgd_opt_optimized(0.01);

            // Instantiate NeuralModelOptimized
            NeuralModelOptimized optimized_model(&adam_opt_optimized, hidden_size, /*epochs=*/500, /*batch_size=*/16); // Reduced epochs
            // If using SGD: NeuralModelOptimized optimized_model(&sgd_opt_optimized, hidden_size, /*epochs=*/500, /*batch_size=*/16);

            // Train and time the Optimized Neural Model
            auto t0_optimized = std::chrono::high_resolution_clock::now();
            optimized_model.fit(data);
            auto t1_optimized = std::chrono::high_resolution_clock::now();
            auto ms_optimized = std::chrono::duration_cast<std::chrono::milliseconds>(t1_optimized - t0_optimized).count();

            // Output the training times for this configuration
            std::cout << std::left << std::setw(25) << ms_standard
                      << std::setw(25) << ms_optimized
                      << "\n";

            // Optional: Add prediction tests here for each model/config if desired,
            // but it might clutter the output for a scalability test.
        }
         std::cout << std::string(80, '-') << "\n"; // Separator line after each data size block
    }

    std::cout << "--- Test Complete ---\n";

    return 0;
}
