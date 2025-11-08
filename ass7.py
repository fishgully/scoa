import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# Params
n_particles, n_iters, k = 20, 50, 4
w, c1, c2 = 0.5, 1.5, 1.5

# Particle class
class Particle:
    def __init__(self):
        self.pos = X[np.random.choice(len(X), k)]
        self.vel = np.zeros_like(self.pos)
        self.best = self.pos.copy()
        self.score = self.eval()

    def eval(self):
        d = np.linalg.norm(X[:, None] - self.pos[None, :], axis=2)
        return np.sum(np.min(d, axis=1)**2)

    def update(self, gbest):
        r1, r2 = np.random.rand(), np.random.rand()
        self.vel = w*self.vel + c1*r1*(self.best - self.pos) + c2*r2*(gbest - self.pos)
        self.pos += self.vel
        s = self.eval()
        if s < self.score: self.score, self.best = s, self.pos.copy()

# Swarm optimization
swarm = [Particle() for _ in range(n_particles)]
gbest = min(swarm, key=lambda p: p.score).best
for _ in range(n_iters):
    for p in swarm: p.update(gbest)
    gbest = min(swarm, key=lambda p: p.score).best

# Plot
labels = np.argmin(np.linalg.norm(X[:, None]-gbest[None,:], axis=2), axis=1)
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(gbest[:,0], gbest[:,1], c='red', marker='x')
plt.title("PSO Clustering")
plt.show()
