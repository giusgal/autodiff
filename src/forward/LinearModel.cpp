#include "LinearModel.h"

int main() {
    auto data = generate_data(100, 5.0, -1.0);  // true model: y = 2x - 1
    LinearModel model(0.01, 200, 20);            // lr, epochs, batch_size
    model.fit(data);

    std::cout << "\nPredictions:\n";
    for (double x = 0.0; x <= 1.0; x += 0.2) {
        std::cout << "x = " << x << ", y_pred = " << model.predict(x) << std::endl;
    }

    return 0;
}