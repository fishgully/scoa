import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Data
X, y = load_iris(return_X_y=True)
# GA setup
POP, GEN, MUT = 10, 5, 0.3

def fitness(ch):
    clf = DecisionTreeClassifier(max_depth=ch[0], min_samples_split=ch[1])
    return cross_val_score(clf, X, y, cv=3).mean()

def mutate(ch):
    if random.random() < MUT: ch[0] = random.randint(1, 20)
    if random.random() < MUT: ch[1] = random.randint(2, 10)
    return ch

# Init population
pop = [[random.randint(1, 20), random.randint(2, 10)] for _ in range(POP)]

for g in range(GEN):
    fit = [fitness(c) for c in pop]
    best = np.argsort(fit)[-2:]
    p1, p2 = pop[best[0]], pop[best[1]]
    pop = [mutate(p1[:1] + p2[1:]), mutate(p2[:1] + p1[1:])] * (POP // 2)
    print(f"Gen {g+1}: Best Acc = {max(fit):.3f}")

# Final best
fit = [fitness(c) for c in pop]
best = pop[np.argmax(fit)]
print("\nBest Params:", best, "Accuracy:", max(fit))
