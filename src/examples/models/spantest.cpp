#include <vector>
#include <span>
#include <cmath>
#include <iostream>

// A minimal DualVar stub; replace with your real DualVar<T>!
template<typename T>
struct DualVar {
    T real_, inf_;
    DualVar(T r=0, T i=0): real_(r), inf_(i) {}
    DualVar operator+(DualVar o) const { return {real_+o.real_, inf_+o.inf_}; }
    DualVar operator*(DualVar o) const {
        return {real_*o.real_, real_*o.inf_ + inf_*o.real_};
    }
};
using DV = DualVar<double>;

// Forward‐pass for one sample, scalar input → scalar output,
// using a single hidden layer of size H, all parameters in
// a flat vector of length 3*H + 1 laid out as:
//   [ W1 (H×1) | b1 (H) | W2 (1×H) | b2 (1) ]
double predict_from_flat(const std::span<const double> params,
                         int H,
                         double x)
{
    // Slicing spans:
    auto w1 = params.subspan(         0,           H     ); // H elements
    auto b1 = params.subspan(         H,           H     ); // next H
    auto w2 = params.subspan(     2*H,           H     );   // next H
    double b2 = params[3*H];                               // last

    // Hidden activations
    std::vector<double> hidden(H);
    for(int i = 0; i < H; ++i) {
        double z = b1[i] + w1[i] * x;
        hidden[i] = std::tanh(z);
    }

    // Output
    double out = b2;
    for(int i = 0; i < H; ++i)
        out += hidden[i] * w2[i];
    return out;
}

// DualVar version
DV predict_from_flat_dual(const std::span<const DV> params,
                          int H,
                          DV x)
{
    auto w1 = params.subspan(         0,           H     );
    auto b1 = params.subspan(         H,           H     );
    auto w2 = params.subspan(     2*H,           H     );
    DV   b2 = params[3*H];

    std::vector<DV> hidden(H);
    for(int i = 0; i < H; ++i) {
        DV z = b1[i] + w1[i] * x;
        // manually inlined tanh derivative
        double zr = std::tanh(z.real_);
        DV dz { zr, (1 - zr*zr) * z.inf_ };
        hidden[i] = dz;
    }

    DV out = b2;
    for(int i = 0; i < H; ++i)
        out = out + hidden[i] * w2[i];
    return out;
}

int main(){
    int H = 3;
    // flat params: [ W1(3), b1(3), W2(3), b2 ]
    std::vector<double> params_flat = {
        0.5, -0.2, 0.1,   // w1
        0.0,  0.1, -0.1,  // b1
        0.3, -0.4, 0.2,   // w2
       -0.05              // b2
    };

    double x = 0.7;
    double y = predict_from_flat(params_flat, H, x);
    std::cout << "predict(double) = " << y << "\n";

    // DualVar example:
    std::vector<DV> params_dual(params_flat.size());
    for(size_t i=0; i<params_flat.size(); ++i)
        params_dual[i] = DV(params_flat[i], 0.0);

    // seed derivative w.r.t x
    DV xdual(x, 1.0);
    DV ydual = predict_from_flat_dual(params_dual, H, xdual);
    std::cout << "predictDual.real = " << ydual.real_
              << ",  predictDual.inf = " << ydual.inf_ << "\n";

    return 0;
}
