import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess the data
data = pd.read_excel("AirQualityUCI.xlsx", na_values=-200)
data.dropna(inplace=True)
X = data.iloc[:, [2, 6, 8, 10, 11, 12, 13, 14]].values
y = data.iloc[:, 5].values

# Standardize the data
X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
hidden_layer = [20]

# Define the MLP class
class MLP:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.weights = [np.random.rand(prev, curr) for prev, curr in zip([input_size] + hidden_layer_sizes, hidden_layer_sizes + [output_size])]

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, X):
        for weight in self.weights:
            X = self.relu(np.dot(X, weight))
        return X

    def predict(self, X):
        return self.forward(X)

# Define the PSO Algorithm
class Particle:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.mlp = MLP(input_size, hidden_layer_sizes, output_size)
        self.position = [w.copy() for w in self.mlp.weights]
        self.velocity = [np.random.rand(*w.shape) * 0.1 for w in self.position]
        self.best_position = self.position
        self.best_error = float("inf")

def pso(X, y, num_particles=15, num_iterations=200):
    particles = [Particle(X.shape[1], hidden_layer, 1) for _ in range(num_particles)]
    global_best_position = None
    global_best_error = float("inf")

    for _ in range(num_iterations):
        for particle in particles:
            y_pred = particle.mlp.predict(X).flatten()
            error = np.mean(np.abs(y - y_pred))

            if error < particle.best_error:
                particle.best_error = error
                particle.best_position = [w.copy() for w in particle.position]

            if error < global_best_error:
                global_best_error = error
                global_best_position = [w.copy() for w in particle.position]

            inertia, cognitive, social = 0.5, 1.5, 1.5
            for i in range(len(particle.position)):
                r1, r2 = np.random.rand(*particle.position[i].shape), np.random.rand(*particle.position[i].shape)
                particle.velocity[i] = (
                    inertia * particle.velocity[i]
                    + cognitive * r1 * (particle.best_position[i] - particle.position[i])
                    + social * r2 * (global_best_position[i] - particle.position[i])
                )
                particle.position[i] += particle.velocity[i]

    return global_best_position

# Cross-validation and performance evaluation
num_folds = 10
fold_size = len(X_scaled) // num_folds
mae_results = []

for fold in range(num_folds):
    test_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
    train_indices = list(set(range(len(X_scaled))) - set(test_indices))

    X_train, y_train = X_scaled[train_indices], y[train_indices]
    X_test, y_test = X_scaled[test_indices], y[test_indices]

    best_weights = pso(X_train, y_train)
    final_mlp = MLP(X_train.shape[1], hidden_layer, 1)
    final_mlp.weights = best_weights

    y_test_pred = final_mlp.predict(X_test).flatten()
    mae = np.mean(np.abs(y_test - y_test_pred))
    mae_results.append(mae)

    print(f"Fold {fold + 1}/{num_folds} - Mean Absolute Error: {mae}")

# Average MAE across all folds
average_mae = np.mean(mae_results)
print(f"Average Mean Absolute Error across all folds: {average_mae}")

# Plot MAE for each fold
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_folds + 1), mae_results, marker="o", linestyle="-", color="b")
plt.title("Mean Absolute Error for Each Fold")
plt.xlabel("Fold Number")
plt.ylabel("Mean Absolute Error")
plt.xticks(range(1, num_folds + 1))
plt.grid()
plt.savefig(f"mae_per_fold_{hidden_layer[0]}_{average_mae:.2f}.png")
plt.show()

# Train on the entire dataset and visualize predictions
best_weights = pso(X_scaled, y)
final_mlp = MLP(X_scaled.shape[1], hidden_layer, 1)
final_mlp.weights = best_weights
y_pred_all = final_mlp.predict(X_scaled).flatten()

# Plot actual vs predicted benzene concentration
plt.figure(figsize=(12, 6))
plt.plot(y, label="Actual Benzene Concentration", color="blue", alpha=0.5)
plt.plot(y_pred_all, label="Predicted Benzene Concentration", color="orange", alpha=0.5)

# Add vertical lines for each fold
for fold in range(1, num_folds):
    plt.axvline(x=fold * fold_size, color="red", linestyle="--", linewidth=1)

plt.title("Benzene Concentration: Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Benzene Concentration (µg/m³)")
plt.legend()
plt.grid()

# Save and show plot
output_path = f"benzene_concentration_{hidden_layer[0]}_{average_mae:.2f}.png"
plt.savefig(output_path)
plt.show()
