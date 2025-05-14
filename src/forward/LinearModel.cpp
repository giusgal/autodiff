#include "LinearModel.h"
#include <chrono>
#include <iostream>

#include "LinearModelParallel.h"

int main() {
    auto data = generate_data(100, 5.0, -2.0);  // true model: y = 2x - 1
    // LinearModel serial execution time
    LinearModel model(0.01, 200, 20);            // lr, epochs, batch_size
    LinearModelParallel modelParallel(0.01, 1000, 20);


    auto t0 = std::chrono::high_resolution_clock::now();
    model.fit(data);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    std::cout << "Serial training took: " << duration1.count() << " microseconds" << std::endl;

    t0 = std::chrono::high_resolution_clock::now();
    modelParallel.fit(data);
    t1 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    std::cout << "Parallel training took: " << duration2.count() << " microseconds" << std::endl;
    std::cout << "Speed Up is " << static_cast<double>(1.0 * duration1 / duration2) << std::endl;

    return 0;
}
