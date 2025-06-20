#include <iostream>
#include <vector>
#include <utility>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm> // Include for std::shuffle

// Include headers for your models and optimizers
#include <NeuralModel.h>
#include <NeuralModelOptimized.h> // Include the header for your optimized model
#include <NeuralModelOpenmp.h>   // Include the header for your OpenMP model
#include <Adam.h>
#include <SGD.h>
#include <IModel.h> // Assuming IModel is the base class

// Define the target function
#define TARGET_FUNCTION(x) (5.0 *(x) - 1.0) // Example: y = 5x - 1

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
    std::vector<int> data_sizes = {200, 600, 1000}; // Added 1000 for better comparison

    int epochs = 500; // Reduced epochs for faster testing
    int batch_size = 16;

    // Output table header
    std::cout << std::left << std::setw(15) << "Data Size"
              << std::setw(15) << "Hidden Size"
              << std::setw(25) << "Standard Model (ms)"
              << std::setw(25) << "Optimized Model (ms)"
              << std::setw(25) << "OpenMP Model (ms)"
              << "\n";
    std::cout << std::string(105, '-') << "\n"; // Adjusted separator line width

    // Loop through data sizes
    for (int data_size : data_sizes) {
        // Generate data for the current size
        auto data = make_data(data_size);

        // Loop through hidden sizes
        for (int hidden_size : hidden_sizes) {
            std::cout << std::left << std::setw(15) << data_size // Print data size for this row
                      << std::setw(15) << hidden_size; // Print hidden size

            // --- Standard Neural Model Training ---
            // Create a new optimizer for each run to ensure a clean state
            Adam adam_opt_standard(0.01);
            NeuralModel standard_model(&adam_opt_standard, hidden_size, epochs, batch_size);

            // Train and time the Standard Neural Model
            auto t0_standard = std::chrono::high_resolution_clock::now();
            standard_model.fit(data);
            auto t1_standard = std::chrono::high_resolution_clock::now();
            auto ms_standard = std::chrono::duration_cast<std::chrono::milliseconds>(t1_standard - t0_standard).count();

            // --- Optimized Neural Model Training ---
            // Create a new optimizer for each run
            Adam adam_opt_optimized(0.01);
            NeuralModelOptimized optimized_model(&adam_opt_optimized, hidden_size, epochs, batch_size);

            // Train and time the Optimized Neural Model
            auto t0_optimized = std::chrono::high_resolution_clock::now();
            optimized_model.fit(data);
            auto t1_optimized = std::chrono::high_resolution_clock::now();
            auto ms_optimized = std::chrono::duration_cast<std::chrono::milliseconds>(t1_optimized - t0_optimized).count();

            // --- OpenMP Neural Model Training ---
            // Create a new optimizer for each run
            Adam adam_opt_openmp(0.01);
            NeuralModelOpenmp openmp_model(&adam_opt_openmp, hidden_size, epochs, batch_size);

            // Train and time the OpenMP Neural Model
            auto t0_openmp = std::chrono::high_resolution_clock::now();
            openmp_model.fit(data);
            auto t1_openmp = std::chrono::high_resolution_clock::now();
            auto ms_openmp = std::chrono::duration_cast<std::chrono::milliseconds>(t1_openmp - t0_openmp).count();

            // Output the training times for this configuration
            std::cout << std::left << std::setw(25) << ms_standard
                      << std::setw(25) << ms_optimized
                      << std::setw(25) << ms_openmp
                      << "\n";

        }
         std::cout << std::string(105, '-') << "\n"; // Separator line after each hidden size block
    }

    std::cout << "--- Test Complete ---\n";

    return 0;
}