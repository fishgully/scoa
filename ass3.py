import numpy as np, matplotlib.pyplot as plt

# --- Triangular MF (safe version) ---
def trimf(x,a,b,c):
    if a==b and x<=b: return 1.0 if x==b else 0.0
    if b==c and x>=b: return 1.0 if x==b else 0.0
    if x<=a or x>=c: return 0.0
    elif x<=b: return (x-a)/(b-a)
    else: return (c-x)/(c-b)

# --- Fuzzy sets ---
err = [lambda x: trimf(x,-30,-30,0),
       lambda x: trimf(x,-10,0,10),
       lambda x: trimf(x,0,30,30)]
derr = [lambda x: trimf(x,-10,-10,0),
        lambda x: trimf(x,-3,0,3),
        lambda x: trimf(x,0,10,10)]
u_sets = [lambda x: trimf(x,-20,-20,-10),
          lambda x: trimf(x,-15,-7,0),
          lambda x: trimf(x,-3,0,3),
          lambda x: trimf(x,0,7,15),
          lambda x: trimf(x,10,20,20)]
u_univ = np.linspace(-20,20,401)

# --- Fuzzy rules ---
def rules(e,de):
    E=[f(e) for f in err]; DE=[f(de) for f in derr]
    return [
        (min(E[0],DE[0]),u_sets[0]),(min(E[0],DE[1]),u_sets[1]),(min(E[0],DE[2]),u_sets[2]),
        (min(E[1],DE[0]),u_sets[1]),(min(E[1],DE[1]),u_sets[2]),(min(E[1],DE[2]),u_sets[3]),
        (min(E[2],DE[0]),u_sets[2]),(min(E[2],DE[1]),u_sets[3]),(min(E[2],DE[2]),u_sets[4])
    ]

# --- Aggregate + defuzzify ---
def defuzz(e,de):
    agg=np.zeros_like(u_univ)
    for s,mf in rules(e,de):
        agg=np.maximum(agg,[min(s,mf(u)) for u in u_univ])
    return np.sum(agg*u_univ)/np.sum(agg) if np.sum(agg)>0 else 0

# --- Simulation ---
angle,target,dt,gain=0,90,0.1,0.05
t_hist,a_hist,u_hist=[],[],[]
for t in np.arange(0,40,dt):
    e,de=target-angle,(target-angle)/dt
    u=np.clip(defuzz(e,de),-20,20)
    angle+=gain*u*dt
    t_hist.append(t); a_hist.append(angle); u_hist.append(u)
    if abs(e)<0.2: break

# --- Plots ---
plt.subplot(1,2,1); plt.plot(t_hist,a_hist); plt.axhline(target,ls='--'); plt.title("Angle vs Time")
plt.subplot(1,2,2); plt.plot(t_hist,u_hist); plt.title("Control vs Time")
plt.show()
