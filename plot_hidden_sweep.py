import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # For generating color sequences

# Define the hidden sizes used in the C++ code
# This should match the hidden sizes used in your C++ sweep
X_hidden = [1, 2, 4, 8, 16]
NUM_NEURAL_MODELS = len(X_hidden) # Number of neural network models to plot

# Attempt to read the CSV file generated by the C++ program
try:
    df = pd.read_csv('model_results_hidden_sweep.csv')
except FileNotFoundError:
    print("Error: 'model_results_hidden_sweep.csv' not found.")
    print("Please ensure your C++ program has run successfully and that")
    print("the CSV file is in the same directory as this Python script.")
    exit() # Exit the script if the file isn't found
except pd.errors.EmptyDataError:
    print("Error: 'model_results_hidden_sweep.csv' is empty.")
    print("Please check the C++ program for any issues during data generation or writing.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the CSV: {e}")
    exit()


print("Columns found in CSV:", df.columns.tolist())

# Extract data for plotting
x_values = df['x'].values
y_true_values = df['y_true'].values
# y_linear_predictions = df['y_linear'].values # Linear model predictions (optional to plot)

# --- Plotting ---
plt.figure(figsize=(14, 9)) # Increased figure size for better clarity with multiple lines

# Zorder management:
# Linear model (if plotted): lowest zorder (e.g., 1)
# Neural Network models: ordered by their index in X_hidden + an offset (e.g., 2 to NUM_NEURAL_MODELS+1)
# True Data points: highest zorder (e.g., NUM_NEURAL_MODELS + 2)
ZORDER_LINEAR = 1
ZORDER_NN_START = 2
ZORDER_TRUE_DATA = NUM_NEURAL_MODELS + ZORDER_NN_START # Place true data on top

# 1. Plot the true data points
plt.scatter(x_values, y_true_values, label='True Data', color='black', s=25, alpha=0.7, zorder=ZORDER_TRUE_DATA)

# 2. Plot the linear model's predictions (optional, uncomment if desired)
# if 'y_linear' in df.columns:
#     y_linear_predictions = df['y_linear'].values
#     plt.plot(x_values, y_linear_predictions, label='Linear Model', color='gray', linestyle='--', linewidth=2, zorder=ZORDER_LINEAR)
# else:
#      print("Warning: Column 'y_linear' was not found in the CSV file. Skipping plot for linear model.")


# 3. Plot predictions from each neural network model
# Generate a colormap for distinct line colors, one color for each hidden size in X_hidden
colors = plt.cm.rainbow(np.linspace(0, 1, NUM_NEURAL_MODELS))

for index, h_size in enumerate(X_hidden):
    column_name = f'y_neural_h{h_size}' # Construct column name, e.g., 'y_neural_h1', 'y_neural_h2', etc.
    if column_name in df.columns:
        y_neural_predictions = df[column_name].values
        plt.plot(x_values, y_neural_predictions,
                 label=f'NN (Hidden: {h_size})',
                 color=colors[index], # Use color based on index
                 linewidth=2.0,
                 alpha=0.9,
                 zorder=index + ZORDER_NN_START) # Assign zorder based on index
    else:
        print(f"Warning: Column '{column_name}' was not found in the CSV file. Skipping plot for hidden size {h_size}.")

# --- Add plot details ---
plt.title('Neural Network Model Approximations with Varying Hidden Layer Sizes', fontsize=16)
plt.xlabel('Input (x)', fontsize=14)
plt.ylabel('Output (y)', fontsize=14)

# Add a legend to identify the lines
# 'loc='best'' tries to place the legend in the least obstructive spot
# 'frameon=True' adds a border around the legend
# 'shadow=True' adds a shadow effect to the legend box
plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)

# Add a grid for easier visual inspection of values
plt.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.6)

# Apply a style for a more professional look (optional)
# plt.style.use('seaborn-v0_8-whitegrid') # Example: using a seaborn style

# Ensure the layout is tight to prevent labels from being cut off
plt.tight_layout()

# --- Save and show the plot ---
try:
    plt.savefig('model_comparison_hidden_sweep.png', dpi=300) # Save with higher resolution
    print("Plot successfully saved as 'model_comparison_hidden_sweep.png'")
except Exception as e:
    print(f"Error encountered while saving the plot: {e}")

plt.show() # Display the plot