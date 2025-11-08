import numpy as np

# ----- Fuzzy Sets -----
X = [1, 2, 3, 4, 5]
A = {1: 0.2, 2: 0.7, 3: 1.0, 4: 0.4, 5: 0.1}
B = {1: 0.6, 2: 0.2, 3: 0.9, 4: 0.3, 5: 0.5}

# ----- Basic Operations -----
union = {x: max(A[x], B[x]) for x in X}
intersection = {x: min(A[x], B[x]) for x in X}
complement_A = {x: 1 - A[x] for x in X}

print("Union:", union)
print("Intersection:", intersection)
print("Complement of A:", complement_A)

# ----- Cartesian Product (Fuzzy Relation) -----
R1 = {(x, y): min(A[x], B[y]) for x in X for y in X}
R2 = {(x, y): min(B[x], A[y]) for x in X for y in X}

print("\nFuzzy Relation R1 (A x B):")
[print(f"{p}: {v}") for p, v in R1.items()]

print("\nFuzzy Relation R2 (B x A):")
[print(f"{p}: {v}") for p, v in R2.items()]

# ----- Max–Min Composition -----
Comp = {(x, z): max(min(R1[(x, y)], R2[(y, z)]) for y in X) for x in X for z in X}

print("\nMax–Min Composition (R1 ∘ R2):")
[print(f"{p}: {v}") for p, v in Comp.items()]
