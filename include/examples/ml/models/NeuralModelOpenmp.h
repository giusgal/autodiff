#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <cmath>
#include <span>

#include <NeuralModel.h>
#include <omp.h>

class NeuralModelOpenmp : public NeuralModel
{
public:
    NeuralModelOpenmp(Optimizer* optimizer,
                const int hidden_size,
                const int epochs = 50,
                const int batch_size = 10): NeuralModel(optimizer, hidden_size, epochs, batch_size){}

    void fit(std::vector<std::pair<double, double>>& data) override
    {
        int num_concurrent_batches = omp_get_max_threads();
        if (num_concurrent_batches <= 0) num_concurrent_batches = 1;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            //the shuffle function still works linearly...
            std::shuffle(data.begin(), data.end(), std::mt19937(epoch));

            //I will work on each meta baches, within each meta batch, i work on minibatches concurrently
            for (size_t i = 0; i < data.size(); i += (size_t)batch_size * num_concurrent_batches)
            {
                //gradient for each minibatch, we will use all the threads available
                std::vector<std::vector<double>> batch_gradients(num_concurrent_batches);
                std::vector<int> actual_samples_in_batch(num_concurrent_batches, 0);

                #pragma omp parallel num_threads(num_concurrent_batches)
                {
                    int thread_id = omp_get_thread_num();
                    size_t current_batch_start_index = i + (size_t)thread_id * batch_size;
                    // i + 0*5,   i+1*5   i+2*5 to start

                    //if we had more threads that its id is > data's size we won't work
                    //we will either set batch_gradients to zero so it won't be calculated or make it empty
                    if (current_batch_start_index < data.size())
                    {
                        auto batch_end_iter = data.begin() + std::min(current_batch_start_index + batch_size, data.size());
                        std::vector<std::pair<double, double>> current_thread_batch(
                            data.begin() + current_batch_start_index,
                            batch_end_iter
                        );

                        if (!current_thread_batch.empty() && !params.empty())
                        {
                            actual_samples_in_batch[thread_id] = current_thread_batch.size();
                            // 'params' is std::vector<double> from NeuralModel
                            // 'gradient' function handles using these doubles with the DualVar lambda
                            batch_gradients[thread_id] = gradient<double>(
                                [&](const std::vector<DualVar<double>>& p_local) { // Lambda uses DualVar
                                    std::span<const DualVar<double>> p_span(p_local.data(), p_local.size());
                                    return loss_func_fused(current_thread_batch, p_span);
                                },
                                params // Pass NeuralModel::params (std::vector<double>)
                            );
                        } else if (!params.empty()) {
                            batch_gradients[thread_id].assign(params.size(), 0.0);
                        } else {
                            batch_gradients[thread_id].clear();
                        }
                    } else if (!params.empty()){
                        batch_gradients[thread_id].assign(params.size(), 0.0);
                    } else {
                         batch_gradients[thread_id].clear();
                    }
                } // End of OpenMP parallel region

                if (params.empty()) continue;

                std::vector<double> aggregated_grad(params.size(), 0.0);
                double total_samples_processed_in_meta_batch = 0;

                for (int k = 0; k < num_concurrent_batches; ++k)
                {
                    if (actual_samples_in_batch[k] > 0 && batch_gradients[k].size() == params.size())
                    {
                        for (size_t j = 0; j < params.size(); ++j)
                        {
                            //weighted average, considering if the last metabatch is < batch_size
                            // we don't want average of gradents, but WEIGHTED AVERAGE like (5,5,5......,3) from 103 data
                            aggregated_grad[j] += batch_gradients[k][j] * static_cast<double>(actual_samples_in_batch[k]);
                        }
                        total_samples_processed_in_meta_batch += actual_samples_in_batch[k];
                    }
                }

                if (total_samples_processed_in_meta_batch > 0)
                {
                    for (size_t j = 0; j < params.size(); ++j)
                    {
                        aggregated_grad[j] /= total_samples_processed_in_meta_batch;
                    }

                    // ** CORRECTED OPTIMIZER UPDATE **
                    // 'params' is std::vector<double>& as expected by optimizer
                    // 'aggregated_grad' is std::vector<double>
                    optimizer->update(params, aggregated_grad);
                    // 'params' is updated in-place by the optimizer. No further steps needed here.
                }
            }
        }
    }

    // Definition of loss_func_fused (assuming autodiff::forward::DualVar is the correct type here)
    DualVar<double> loss_func_fused(const std::vector<std::pair<double, double>>& batch,
                              const std::span<const DualVar<double>> p_dual
    ) const {
        double sum_real_loss = 0;
        double sum_inf_loss = 0;

        // Ensure hidden_size is accessible (member of NeuralModel or NeuralModelOpenmp)
        auto w1 = p_dual.subspan(0, this->hidden_size); // Use this->hidden_size for clarity if it's a member
        auto b1 = p_dual.subspan(this->hidden_size, this->hidden_size);
        auto w2 = p_dual.subspan(2 * this->hidden_size, this->hidden_size);
        DualVar<double> b2 = p_dual.subspan(3 * this->hidden_size, 1).back();

        #pragma omp parallel for reduction(+:sum_real_loss) reduction(+:sum_inf_loss) default(none) \
            shared(batch, w1, b1, w2, b2, hidden_size) // Add hidden_size to shared if accessed directly
                                                        // and not passed via p_dual structure indirectly.
                                                        // If hidden_size is a const member, it's fine.
        for (size_t i = 0; i < batch.size(); ++i)
        {
            const auto& [x_, y_] = batch[i];
            DualVar<double> x(x_, 0.0); // Assuming DualVar is autodiff::forward::DualVar<double>
            DualVar<double> y(y_, 0.0);
            std::vector<DualVar<double>> hidden(this->hidden_size); // Use this->hidden_size
            for (int j = 0; j < this->hidden_size; j++)
            {
                hidden[j] = tanh(w1[j] * x + b1[j]);
            }
            DualVar<double> out = b2;
            for (int j = 0; j < this->hidden_size; j++)
            {
                out = out + hidden[j] * w2[j];  //there is no += operator implemented in dualvar, but it is ok
            }
            DualVar<double> diff = out - y;
            sum_real_loss += diff.getReal() * diff.getReal();
            sum_inf_loss += 2 * diff.getReal() * diff.getInf();
            //always using the algebra of dual numbers (a + be) * (c + de)
            // = a*c and a*de + be*c,  be*de = 0..
            //a*de + be*c = 2*a*de 
        }

        // Avoid division by zero if batch is empty, though current logic
        // for `actual_samples_in_batch` should prevent this path for empty batches
        if (batch.empty()) {
            return DualVar<double>(0.0, 0.0);
        }
        return DualVar<double>(sum_real_loss / batch.size(), sum_inf_loss / batch.size());
    }
};
