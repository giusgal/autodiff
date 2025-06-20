"""
USAGE:
function-generator.py --input-dim X --output-dim Y
    --complexity N --expr-length M --seed S --jacobian-density D

    --complexity controls the number of iterations that each function output will perform
    --expr-length controls the length of the random chains of operations that are generated
    --jacobian-density is a float between 0 and 1
"""
import random
import argparse
from typing import List, Tuple


class FunctionGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        # Available mathematical functions with safeguards to avoid NaNs
        self.unary_functions = ['sin', 'cos', 'sqrt_safe', 'abs', 'log_safe', 'exp_safe']
        self.binary_operations = ['+', '-', '*']

        # Small coefficients to prevent explosion
        self.max_coeff = 0.3
        self.min_coeff = 0.05

    def generate_expression(self, input_dim: int, expr_length: int, active_vars: List[int] = None) -> str:
        if active_vars is None:
            active_vars = list(range(input_dim))

        terms = []

        for _ in range(expr_length):
            # Operation type
            op_type = self.rng.choice(['unary', 'binary', 'product'])
            ""

            if op_type == 'unary':
                # Choose unary function
                func = self.rng.choice(self.unary_functions)
                # Choose variable on which to apply the function
                var_idx = self.rng.choice(active_vars)
                # Random small coefficient
                coeff = self.rng.uniform(self.min_coeff, self.max_coeff)

                # All the following functions are multiplied by a factor depending on j, the iterator
                # This is to prevent the compiler from optimizing away most of the calculations during benchmarks
                if func == 'sqrt_safe':
                    # Always positive argument
                    terms.append(f"sqrt(x[{var_idx}] * x[{var_idx}] + 0.01) * static_cast<double>(j % 10) * {coeff:.3f}")
                elif func == 'log_safe':
                    # Always positive argument
                    terms.append(f"log(x[{var_idx}] * x[{var_idx}] + 1.0) * static_cast<double>(j % 10) * {coeff:.3f}")
                elif func == 'exp_safe':
                    # Shrink exponent
                    terms.append(f"exp(x[{var_idx}] * 0.1) * static_cast<double>(j % 10) * {coeff:.3f}")
                elif func in ['sin', 'cos', 'tanh', 'atan', 'abs']:
                    terms.append(f"{func}(x[{var_idx}]) * static_cast<double>(j % 10) * {coeff:.3f}")

            elif op_type == 'binary':
                # Binary operations with very small coefficients
                var1 = self.rng.choice(active_vars)
                var2 = self.rng.choice(active_vars)
                op = self.rng.choice(self.binary_operations)
                coeff = self.rng.uniform(self.min_coeff, self.max_coeff)

                if op == '*':
                    # Small coefficient for multiplication
                    coeff *= 0.05
                    terms.append(f"(x[{var1}] {op} x[{var2}]) * static_cast<double>(j % 10) * {coeff:.4f}")
                else:
                    terms.append(f"(x[{var1}] {op} x[{var2}]) * static_cast<double>(j % 10) * {coeff:.3f}")

            else:
                # Limited products with small coefficients
                num_vars = self.rng.randint(2, min(3, len(active_vars)))
                vars_used = self.rng.sample(active_vars, num_vars)
                coeff = self.rng.uniform(self.min_coeff, self.max_coeff) / (num_vars ** 2)
                product = " * ".join([f"x[{i}]" for i in vars_used])
                terms.append(f"({product}) * {coeff:.4f}")
            terms.append(f"static_cast<double>(j % 10) * {coeff}")
        return " + ".join(terms)

    def select_active_variables(self, input_dim: int, density: float) -> List[int]:
        num_active = max(1, int(input_dim * density))
        return self.rng.sample(range(input_dim), num_active)


def generate_header_file(input_dim: int, output_dim: int, complexity: int, expr_length: int,
                         output_file: str, seed: int = 42, jacobian_density: float = 1.0):

    generator = FunctionGenerator(seed)

    # Generate expressions for each output with controlled density
    expressions = []
    active_vars_per_output = []

    for i in range(output_dim):
        active_vars = generator.select_active_variables(input_dim, jacobian_density)
        active_vars_per_output.append(active_vars)
        expr = generator.generate_expression(input_dim, expr_length, active_vars)
        expressions.append(expr)

    # Generate clean header without runtime checks
    header_content = f'''#pragma once
#include <Eigen/Dense>
#include <cmath>
#include "../../autodiff/forward/CudaSupport.hpp"
#include "../../autodiff/forward/autodiff.hpp"
#include "Jacobian.hpp"

using dv = autodiff::forward::DualVar<double>;
using dvec = Eigen::Matrix<dv, Eigen::Dynamic, 1>;

namespace testfun {{
    constexpr int input_dim = {input_dim};
    constexpr int output_dim = {output_dim};
    constexpr int complexity = {complexity};

    dvec test_fun(const dvec &x) {{
        dvec res({output_dim});
        dv acc;
'''

    # Add each function output - simple and stable
    for i, expr in enumerate(expressions):
        header_content += f'''        
        // Function output {i} (depends on variables: {sorted(active_vars_per_output[i])})
        acc = 0;
        for(int j = 0; j < complexity; j++){{
            acc = acc + ({expr}) / static_cast<double>(complexity);
        }}
        res[{i}] = acc;
'''

    header_content += '''        
        return res;
    }

    #ifdef USE_CUDA

    CUDA_DEVICE dv cu_f0(const dvec &x, const int y) { 
        dv acc = 0;
        switch (y)
        {
'''

    # Add CUDA switch cases with same stability
    for i, expr in enumerate(expressions):
        header_content += f'''        case {i}:
            for(int j = 0; j < complexity; j++){{
                acc = acc + ({expr}) / static_cast<double>(complexity);
            }}
            break;
'''

    header_content += '''        default:
            break;
        }
        return acc;
    };

    newton::CudaFunctionWrapper<double> createcudafn() {
        newton::CudaFunctionWrapper<double> cudafun;
        cudafun.register_fn_host<cu_f0>();
        return cudafun;
    }
    #endif
} // namespace testfun
'''

    # Write to file
    with open(output_file, 'w') as f:
        f.write(header_content)


    print(f"Generated {output_file} with:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Expr length: {expr_length}")
    print(f"  Complexity: {complexity}")
    print(f"  Random seed: {seed}")
    print(f"  Target Jacobian density: {jacobian_density:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Generate example-functions.hpp files')
    parser.add_argument('--input-dim', type=int, default=3,
                        help='Input vector dimension (default: 3)')
    parser.add_argument('--output-dim', type=int, default=2,
                        help='Number of function outputs (default: 2)')
    parser.add_argument('--complexity', type=int, default=5,
                        help='Function complexity (number of terms per loop iteration, default: 5)')
    parser.add_argument('--expr-length', type=int, default=5,
                        help='Expression length (number of terms per expression, default: 5)')
    parser.add_argument('--jacobian-density', type=float, default=1.0,
                        help='Jacobian density (0.0-1.0, fraction of non-zero entries, default: 1.0)')
    parser.add_argument('--output', type=str, default='example-functions.hpp',
                        help='Output file name (default: example-functions.hpp)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Validate jacobian density
    if not 0.0 < args.jacobian_density <= 1.0:
        parser.error("jacobian-density must be between 0.0 and 1.0")

    generate_header_file(args.input_dim, args.output_dim, args.complexity, args.expr_length,
                         args.output, args.seed, args.jacobian_density)


if __name__ == '__main__':
    main()