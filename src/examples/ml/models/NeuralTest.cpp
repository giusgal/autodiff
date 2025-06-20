#include <iostream>
#include <vector>
#include <utility>
#include <chrono>

#include <NeuralModel.h>
#include <Adam.h>         // Corrected include for Adam.h
#include <optimizer/SGD.h>          // Corrected include for SGD.h
#include <models/IModel.h>          // Corrected include for IModel.h

#define TARGET_FUNCTION(x) (5.0 *(x) - 1.0) // Example: y = 3x^2 + 2x - 5
#define TARGET_FUNCTION1(x) (3.0 * (x) * (x) + 2.0 * (x) - 5.0) // Example: y = 3x^2 + 2x - 5


// A simple helper to generate linear data y = 2xx + 3x - 5
static std::vector<std::pair<double,double>> make_data(int N) {
    std::vector<std::pair<double,double>> data;
    std::mt19937 rng(123);
    std::normal_distribution<double> noise(0.0, 0.05);
    for(int i = 0; i < N; ++i){
        double x = (i / double(N - 1)) * 2.0 - 1.0;  // in [−1, +1]
        double y = TARGET_FUNCTION(x);  // noise(rng);
        data.emplace_back(x,y);
    }
    return data;
}

int main(){
    // 1) Generate dataset
    auto data = make_data(200);

    // 2) Create optimizers
    Adam adam_opt(0.01);
    //SGD  sgd_opt(0.01);

    // 3) Instantiate NeuralModel (hidden_size = 8)
    NeuralModel model(&adam_opt, /*hidden_size=*/3, /*epochs=*/1000, /*batch_size=*/16);

    // 4) Train and time it
    auto t0 = std::chrono::high_resolution_clock::now();
    model.fit(data);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "Training completed in " << ms << " ms\n";

    // 5) Inspect parameters
    model.print_parameters();
    // 6) Test predictions
    std::cout << "\nSample predictions:\n";
    for (double x : {-1.0, -0.5, 0.0, 0.5, 1.0}) {
        double y_pred = model.predict(x);
        double y_true = TARGET_FUNCTION(x);
        std::cout << "x=" << x
                  << "  pred=" << y_pred
                  << "  true=" << y_true
                  << "\n";
    }

    return 0;
}
