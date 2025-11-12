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



# 🧠 Let’s break it down carefully:

# Input: positions → each particle’s position represents cluster centroids.
# (Each particle = possible clustering solution)

# Line	Explanation
# n_particles = positions.shape[0]	Number of particles (solutions in the swarm)
# cost = np.zeros(n_particles)	Initialize cost array for all particles
# for i in range(n_particles):	For every particle, calculate its fitness (cost)
# centroids = positions[i].reshape(k, X.shape[1])	Reshape the flat position vector into k centroids (each has 3 features)
# d = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)	Compute distance between each data point and each centroid
# np.min(d, axis=1)	Find nearest centroid for each point
# np.sum(np.min(d, axis=1)**2)	Sum of squared distances (SSE — same as K-Means objective)
# return cost	Return all particles’ costs

# ✅ Goal of PSO: minimize this cost → minimize distance between points and their assigned cluster centers.

# 🐦 Initialize the PSO Optimizer
# options = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
# optimizer = GlobalBestPSO(n_particles=20, dimensions=k*X.shape[1], options=options)


# Explanation:

# PSO Parameters:

# n_particles=20 → 20 possible clustering solutions (particles)

# dimensions=k*X.shape[1] → each particle has 4 clusters × 3 features = 12 dimensions

# options:

# c1=1.5 → cognitive coefficient (how much a particle trusts itself)

# c2=1.5 → social coefficient (how much it trusts the swarm’s best)

# w=0.5 → inertia weight (controls momentum from previous velocity)

# 💡 The optimizer moves each particle in search of the global best centroids.

# 🧭 Run the Optimization
# best_cost, best_pos = optimizer.optimize(cost_function, iters=50)
# best_centroids = best_pos.reshape(k, X.shape[1])


# Explanation:

# Runs PSO for 50 iterations

# Each iteration updates particle positions to minimize cost function

# Returns:

# best_cost → lowest clustering error found

# best_pos → position (centroids) of the best solution

# Reshape that best position into centroid coordinates (4×3)

# ✅ best_centroids now contains final cluster centers found by PSO.

# 🏷️ Assign Each Data Point to a Cluster
# labels = np.argmin(np.linalg.norm(X[:, None] - best_centroids[None, :], axis=2), axis=1)


# Explanation:

# Calculates distances from each data point to each centroid

# Finds the index of the nearest centroid → gives the cluster label

# labels = array of integers (0 to 3)

# 💡 Same as K-Means’ assignment step but using PSO-optimized centroids.

