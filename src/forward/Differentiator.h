// Differentiator.h
#ifndef __DIFFERENTIATOR__H__
#define __DIFFERENTIATOR__H__

#include <vector>
#include <functional>
#include "DualVar.h"

namespace autodiff {
    namespace forward {

        DualVar<double> derivative(std::function<DualVar<double>(DualVar<double>)>f, double x0){
            DualVar<double> res = f(DualVar<double>(x0, 1.0));
            return res;
        }

        std::vector<double> gradient(std::function<DualVar<double>(std::vector<DualVar<double>>)>f,
            std::vector<double> x) {

            std::vector<DualVar<double>> xd;
            std::vector<double> res;

            xd.reserve(size(x));
            res.reserve(size(x));

            for(int i = 0; i < size(x); i++){
                xd.push_back(DualVar<double>(x[i], 0.0));
            }

            for(int i = 0; i < size(x); i++){
                xd[i].setInf(1.0);
                res.push_back(f(xd).getInf());
                xd[i].setInf(0.0);
            }

            return res;
        }

    } // namespace forward
} // namespace autodiff

#endif // __DIFFERENTIATOR__H__
