import pandas as pd
import matplotlib.pyplot as plt
import sys

# Get the hidden size from command line argument
if len(sys.argv) < 2:
    print("Usage: python plot_approximation.py <hidden_size>")
    sys.exit(1)

hidden_size = sys.argv[1]
file_path = f'prediction_data_h{hidden_size}.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: {file_path} not found. Please ensure the C++ program generated it.")
    sys.exit(1)

plt.figure(figsize=(12, 7))
plt.plot(df['x'], df['y_true'], label='True Function', color='black', linestyle='--', linewidth=2)
plt.plot(df['x'], df['y_standard_pred'], label='Standard Model Prediction', alpha=0.8)
plt.plot(df['x'], df['y_optimized_pred'], label='Optimized Model Prediction', alpha=0.8)
plt.plot(df['x'], df['y_openmp_pred'], label='OpenMP Model Prediction', alpha=0.8)

plt.title(f'Neural Model Approximation for Hidden Size: {hidden_size}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# --- Save and show the plot ---
try:
    plt.savefig('three_models_verification.png', dpi=300) # Save with higher resolution
    print("Plot successfully saved as 'three_models_verification.png'")
except Exception as e:
    print(f"Error encountered while saving the plot: {e}")




plt.show()

print(f"Approximation plot for hidden size {hidden_size} displayed.")
