#ifndef LINEARMODELPARALLEL_H
#define LINEARMODELPARALLEL_H

#include "LinearModel.h"


class LinearModelParallel : public LinearModel{
    public:

    using LinearModel::LinearModel;

    void fit(std::vector<std::pair<double, double>>& data) override
{
    std::vector<double> params = { w, b };
    std::mt19937 rng(0);

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(data.begin(), data.end(), rng);

        // Number of mini‐batches this epoch
        int num_batches = (data.size() + batch_size - 1) / batch_size;

        // Accumulator for summed gradients across all batches
        std::vector<double> grad_sum(params.size(), 0.0);

#pragma omp parallel
        {
            // Each thread keeps its own local copy to reduce into
            std::vector<double> local_grad(params.size(), 0.0);

            #pragma omp for
            for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
            {
                int start = batch_idx * batch_size;
                int end   = std::min(start + batch_size, (int)data.size());

                std::vector<std::pair<double, double>> batch(
                    data.begin() + start,
                    data.begin() + end
                );

                // Compute this batch’s gradient
                auto grad = gradient(
                    [&](const std::vector<DualVar<double>>& p) {
                        return loss_func(batch, p[0], p[1]);
                    },
                    params
                );

                // Accumulate into thread-local gradient
                for (size_t j = 0; j < params.size(); ++j)
                    local_grad[j] += grad[j];
            }

            // Safely   add thread-local results into the shared sum
            #pragma omp critical
            for (size_t j = 0; j < params.size(); ++j)
                grad_sum[j] += local_grad[j];
        }

        // Average gradient over batches and update parameters
        for (size_t j = 0; j < params.size(); ++j)
            grad_sum[j] /= num_batches;
        optimizer->update(params, grad_sum);
    }

    // Commit final parameters
    w = params[0];
    b = params[1];
    std::cout << "| w: " << w << " | b: " << b << std::endl;
}

};

#endif //LINEARMODELPARALLEL_H
