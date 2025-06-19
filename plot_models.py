
import pandas as pd
import matplotlib.pyplot as plt

# Read the results CSV
df = pd.read_csv('model_results.csv')

print("Columns found in CSV:", df.columns.tolist())


# Create a grid of x values for smooth curves
x = df['x'].values
y_true = df['y_true'].values
y_lin = df['y_linear'].values
y_nn = df['y_neural'].values

plt.figure(figsize=(8, 6))
# Scatter of true data
plt.scatter(x, y_true, label='True data', color='black', s=20)
# Linear model curve
plt.plot(x, y_lin, label='Linear Model', color='blue')
# Neural model curve
plt.plot(x, y_nn, label='Neural Model', color='red')

plt.title('Model Approximations vs True Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
