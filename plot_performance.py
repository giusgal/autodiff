import pandas as pd
import matplotlib.pyplot as plt

file_path = 'performance_data.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: {file_path} not found. Please ensure the C++ program generated it.")
    exit(1)

plt.figure(figsize=(12, 7))
plt.plot(df['HiddenSize'], df['StandardModel_ms'], label='Standard Model', marker='o')
plt.plot(df['HiddenSize'], df['OptimizedModel_ms'], label='Optimized Model', marker='o')
plt.plot(df['HiddenSize'], df['OpenMPModel_ms'], label='OpenMP Model', marker='o')

plt.title('Neural Model Training Time vs. Hidden Layer Size')
plt.xlabel('Hidden Layer Size')
plt.ylabel('Training Time (ms)')
plt.xscale('log', base=2) # Use log scale for hidden size if they are powers of 2
plt.xticks(df['HiddenSize'], labels=df['HiddenSize']) # Ensure all hidden sizes are shown as ticks
plt.legend()
plt.grid(True, which="both", ls="--", c='0.7')

# --- Save and show the plot ---
try:
    plt.savefig('model_performance.png', dpi=300) # Save with higher resolution
    print("Plot successfully saved as 'model_performance.png'")
except Exception as e:
    print(f"Error encountered while saving the plot: {e}")



plt.show()

print("Performance plot displayed.")
