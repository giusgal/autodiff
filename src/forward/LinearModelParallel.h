//
// Created by sheldon on 5/13/25.
//

#ifndef LINEARMODELPARALLEL_H
#define LINEARMODELPARALLEL_H
class LinearModelParallel{
private:
    double w;
    double b;
    double lr;
    int epochs;
    int batch_size;

    DualVar<double> loss_func(const std::vector<std::pair<double, double>>& batch,
                                DualVar<double> w_,
                                DualVar<double> b_)
    {
        double real_accum = 0.0;
        double inf_accum = 0.0;
        for (const auto& [x, y] : batch)
        {
            DualVar x_dual(x, 0.0);
            DualVar y_dual(y, 0.0);
            DualVar<double> y_pred = w_ * x_dual + b_;
            DualVar<double> diff = y_pred - y_dual;
            //this diff * diff  is the loss squared, we want to know dloss/dw and dloss/db
            //the derivative of this would be 2*loss * dloss/dw and 2*loss * dloss/db
            real_accum += (diff * diff).getReal();
            inf_accum += (diff * diff).getInf();
        }
        return DualVar<double>(real_accum / batch.size(), inf_accum / batch.size());
    }

    public:
    LinearModelParallel(double lr = 0.01, int epochs = 50, int batch_size = 10)
        :w(0.0), b(0.0), lr(lr), epochs(epochs), batch_size(batch_size){}

    void fit(std::vector<std::pair<double, double>>& data)
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
            params[j] -= lr * (grad_sum[j] / num_batches);
    }

    // Commit final parameters
    w = params[0];
    b = params[1];
    std::cout << "| w: " << w << " | b: " << b << std::endl;
}

    double predict(double x) const
    {
        return w * x + b;
    }

    std::pair<double, double> get_params() const{
        return {w, b};
    }

};

#endif //LINEARMODELPARALLEL_H
