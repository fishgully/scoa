# import numpy as np, matplotlib.pyplot as plt

# # --- Triangular MF (safe version) ---
# def trimf(x,a,b,c):
#     if a==b and x<=b: return 1.0 if x==b else 0.0
#     if b==c and x>=b: return 1.0 if x==b else 0.0
#     if x<=a or x>=c: return 0.0
#     elif x<=b: return (x-a)/(b-a)
#     else: return (c-x)/(c-b)

# # --- Fuzzy sets ---
# err = [lambda x: trimf(x,-30,-30,0),
#        lambda x: trimf(x,-10,0,10),
#        lambda x: trimf(x,0,30,30)]
# derr = [lambda x: trimf(x,-10,-10,0),
#         lambda x: trimf(x,-3,0,3),
#         lambda x: trimf(x,0,10,10)]
# u_sets = [lambda x: trimf(x,-20,-20,-10),
#           lambda x: trimf(x,-15,-7,0),
#           lambda x: trimf(x,-3,0,3),
#           lambda x: trimf(x,0,7,15),
#           lambda x: trimf(x,10,20,20)]
# u_univ = np.linspace(-20,20,401)

# # --- Fuzzy rules ---
# def rules(e,de):
#     E=[f(e) for f in err]; DE=[f(de) for f in derr]
#     return [
#         (min(E[0],DE[0]),u_sets[0]),(min(E[0],DE[1]),u_sets[1]),(min(E[0],DE[2]),u_sets[2]),
#         (min(E[1],DE[0]),u_sets[1]),(min(E[1],DE[1]),u_sets[2]),(min(E[1],DE[2]),u_sets[3]),
#         (min(E[2],DE[0]),u_sets[2]),(min(E[2],DE[1]),u_sets[3]),(min(E[2],DE[2]),u_sets[4])
#     ]

# # --- Aggregate + defuzzify ---
# def defuzz(e,de):
#     agg=np.zeros_like(u_univ)
#     for s,mf in rules(e,de):
#         agg=np.maximum(agg,[min(s,mf(u)) for u in u_univ])
#     return np.sum(agg*u_univ)/np.sum(agg) if np.sum(agg)>0 else 0

# # --- Simulation ---
# angle,target,dt,gain=0,90,0.1,0.05
# t_hist,a_hist,u_hist=[],[],[]
# for t in np.arange(0,40,dt):
#     e,de=target-angle,(target-angle)/dt
#     u=np.clip(defuzz(e,de),-20,20)
#     angle+=gain*u*dt
#     t_hist.append(t); a_hist.append(angle); u_hist.append(u)
#     if abs(e)<0.2: break

# # --- Plots ---
# plt.subplot(1,2,1); plt.plot(t_hist,a_hist); plt.axhline(target,ls='--'); plt.title("Angle vs Time")
# plt.subplot(1,2,2); plt.plot(t_hist,u_hist); plt.title("Control vs Time")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Triangular Membership Function
# ---------------------------------------------------------
def trim(x, a, b, c):
    if x <= a or x >= c:
        return 0
    if x == b:
        return 1
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


# ---------------------------------------------------------
# Membership Functions for Error
# ---------------------------------------------------------
def error_negative(x): return trim(x, -30, -15, 0)
def error_zero(x):     return trim(x, -10, 0, 10)
def error_positive(x): return trim(x, 0, 15, 30)

# ---------------------------------------------------------
# Membership Functions for ΔError
# ---------------------------------------------------------
def derror_negative(x): return trim(x, -10, -5, 0)
def derror_zero(x):     return trim(x, -3,  0, 3)
def derror_positive(x): return trim(x, 0,   5, 10)

# ---------------------------------------------------------
# Output Universe
# ---------------------------------------------------------
u_universe = np.linspace(-20, 20, 401)

# Output MFs
def sn(x): return trim(x, -20, -15, -10)   # strong negative
def zero(x): return trim(x, -3, 0, 3)
def sp(x): return trim(x, 10, 15, 20)      # strong positive
def wn(x): return trim(x, -15, -7, 0)      # weak negative
def wp(x): return trim(x, 0, 7, 15)        # weak positive


# ---------------------------------------------------------
# Rule Base
# ---------------------------------------------------------
rules = [
    ("neg",  "any",  wp),
    ("pos",  "any",  wn),
    ("zero", "neg",  sp),
    ("zero", "pos",  sn),
    ("zero", "zero", zero)
]


# ---------------------------------------------------------
# Rule Evaluation
# ---------------------------------------------------------
def eva_rules(error, derror):
    E = {
        "neg": error_negative(error),
        "pos": error_positive(error),
        "zero": error_zero(error)
    }

    DE = {
        "neg": derror_negative(derror),
        "pos": derror_positive(derror),
        "zero": derror_zero(derror)
    }

    fired_rules = []

    for e, d, c in rules:
        de_strength = max(DE.values()) if d == "any" else DE[d]
        rule_strength = min(E[e], de_strength)
        fired_rules.append((rule_strength, c))

    return fired_rules


# ---------------------------------------------------------
# Aggregation
# ---------------------------------------------------------
def aggr_rules(fired):
    output = np.zeros_like(u_universe)

    for strength, mf in fired:
        clipped = np.array([min(strength, mf(u)) for u in u_universe])
        output = np.maximum(output, clipped)

    return output


# ---------------------------------------------------------
# Defuzzification
# ---------------------------------------------------------
def defuzzify(agg_output):
    num = np.sum(u_universe * agg_output)
    den = np.sum(agg_output)
    return 0 if den == 0 else num / den


# ---------------------------------------------------------
# Simulation
# ---------------------------------------------------------
target_angle = 90
current_angle = 0

dt = 0.1
gain = 0.1
steps = 400

time_log = []
angle_log = []
control_log = []

prev_err = target_angle - current_angle

for i in range(steps):
    err = target_angle - current_angle
    derr = (err - prev_err) / dt

    fired = eva_rules(err, derr)
    agg = aggr_rules(fired)
    control_signal = defuzzify(agg)

    # Update system
    current_angle += control_signal * gain * dt

    # Store logs
    time_log.append(i * dt)
    angle_log.append(current_angle)
    control_log.append(control_signal)

    prev_err = err


# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(time_log, angle_log)
plt.title("Angle vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.axhline(target_angle, color='r', linestyle='--')

plt.subplot(1,2,2)
plt.plot(time_log, control_log)
plt.title("Control Signal vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Control Output")

plt.tight_layout()
plt.show()