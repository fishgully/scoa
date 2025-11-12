from pyswarms.single.global_best import GlobalBestPSO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load dataset ---
df = pd.read_csv(r"C:\Users\varun\Downloads\SCOA_A7.csv")

# Use numeric columns for clustering
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values
k = 4  # number of clusters

# --- Cost function for PSO clustering ---
def cost_function(positions):
    n_particles = positions.shape[0]
    cost = np.zeros(n_particles)
    for i in range(n_particles):
        centroids = positions[i].reshape(k, X.shape[1])
        d = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        cost[i] = np.sum(np.min(d, axis=1)**2)
    return cost

# --- Initialize PSO optimizer ---
options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
optimizer = GlobalBestPSO(n_particles=20, dimensions=k*X.shape[1], options=options)

# --- Optimize ---
best_cost, best_pos = optimizer.optimize(cost_function, iters=50)
best_centroids = best_pos.reshape(k, X.shape[1])

# --- Assign cluster labels ---
labels = np.argmin(np.linalg.norm(X[:, None]-best_centroids[None,:], axis=2), axis=1)

# --- Visualization (2D using two features) ---
plt.figure(figsize=(8,6))
plt.scatter(X[:,1], X[:,2], c=labels, cmap='viridis', s=50)
plt.scatter(best_centroids[:,1], best_centroids[:,2], c='red', marker='x', s=200)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("PSO Clustering on Customer Dataset")
plt.show()
