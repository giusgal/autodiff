#include <iostream>
#include <vector>
#include <chrono>

#include "LinearModel.h"
#include "Adam.h"
#include "Optimizer.h"
#include "SGD.h"
#include "SGDWithMomentum.h"


// Function to test a specific optimizer with the LinearModel
void test_optimizer(std::vector<std::pair<double, double>>& data, Optimizer* optimizer, const std::string& optimizer_name, const int epochs, const int batch_size) {
    std::cout << "--- Testing " << optimizer_name << " ---" << std::endl;

    // Create a LinearModel with the given optimizer
    LinearModel model(optimizer, epochs, batch_size);

    // Measure training time
    auto t0 = std::chrono::high_resolution_clock::now();
    model.fit(data); // Assuming fit takes std::vector<DataPoint>
    auto t1 = std::chrono::high_resolution_clock::now();

    // Calculate and print duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    std::cout << optimizer_name << " training took: " << duration.count() << " microseconds" << std::endl;

    model.print_parameters();

    std::cout << "---------------------------------" << std::endl;
}

int main() {
    // Generate synthetic data
    auto data = generate_data(100, 5.0, -2.0); // true model: y = 5x - 2

    double lr = 0.001;
    // Create optimizer instances
    Optimizer* optimizerSGD = new SGD(lr);
    Optimizer* optimizerAdam = new Adam(lr);
    Optimizer* optimizerSGDWithMo = new SGDWithMomentum(lr, 0.9, 2);

    // Define training parameters
    int epochs = 200;
    int batch_size = 20;

    // Test each optimizer
    test_optimizer(data, optimizerSGD, "SGD", epochs, batch_size);
    test_optimizer(data, optimizerAdam, "Adam", epochs, batch_size);
    test_optimizer(data, optimizerSGDWithMo, "SGDWithMomentum", epochs, batch_size);

    // Clean up allocated optimizers to prevent memory leaks
    delete optimizerSGD;
    delete optimizerAdam;
    delete optimizerSGDWithMo;

    return 0;
}