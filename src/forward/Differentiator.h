// Differentiator.h
#ifndef __DIFFERENTIATOR__H__
#define __DIFFERENTIATOR__H__

#include <vector>
#include <functional>
#include "DualVar.h"

namespace autodiff {
    namespace forward {

        // Single-variable derivative (forward‐mode)
        inline DualVar<double> derivative(
            std::function<DualVar<double>(DualVar<double>)> f,
            double x0)
        {
            return f(DualVar<double>(x0, 1.0));
        }

        // n‐dimensional gradient
        inline std::vector<double> gradient(
            std::function<DualVar<double>(std::vector<DualVar<double>>)> f,
            std::vector<double> x)
        {
            std::vector<DualVar<double>> xd;
            std::vector<double> res;
            xd.reserve(x.size());
            res.reserve(x.size());

            for (size_t i = 0; i < x.size(); ++i)
                xd.push_back(DualVar<double>(x[i], 0.0));

            for (size_t i = 0; i < x.size(); ++i) {
                xd[i].setInf(1.0);
                res.push_back(f(xd).getInf());
                xd[i].setInf(0.0);
            }
            return res;
        }

    } // namespace forward
} // namespace autodiff

#endif // __DIFFERENTIATOR__H__
