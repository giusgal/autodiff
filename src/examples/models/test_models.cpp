#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <utility>
#include "LinearModel.h"
#include "NeuralModel.h"
#include "../optimizer/Adam.h"
#include "../optimizer/SGD.h"
#include "../optimizer/SGDWithMomentum.h"

int main() {
    // Generate data
    std::vector<std::pair<double,double>> data;
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.1);
    int N = 200;
    for(int i=0;i<N;i++){
        double x = (i/(double)(N-1)) + 2.0 - 1.0;
        double y =
            std::sin(2 * 3.14159265358979323846 * x) + 0.3 * x * x * x + noise(rng);
        data.emplace_back(x,y);
    }

    // Train linear model
    Adam adam_lin(0.03);
    LinearModel lin(&adam_lin, 2000, 16);
    lin.fit(data);
    auto p_lin = lin.get_params();

    // Train neural model
    Adam adam_nn(0.03);
    NeuralModel nn(&adam_nn, 16, 2000, 16);
    nn.fit(data);

    // Open CSV
    std::ofstream out("model_results.csv");
    out << "x,y_true,y_linear,y_neural\n";

    std::sort(data.begin(), data.end(), [](auto &a, auto &b){ return a.first < b.first; });
    for(auto &pt: data){
        double x = pt.first;
        double y = pt.second;
        double ylin = lin.predict(x);
        double ynn  = nn.predict(x);
        out << x << "," << y << "," << ylin << "," << ynn << "\n";
    }
    out.close();
    return 0;
}