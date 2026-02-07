import numpy as np
import matplotlib.pyplot as plt

# Standalone impedance-coupling tracking test (translation + rotation)
# We simulate a "rod tip" DOF with mass m and inertia I, driven by an impedance coupling
# to a target trajectory (x_t, v_t) and (theta_t, omega_t).
#
# Core formulas (per step):
#   F   = kp*(x_t - x) + kd*(v_t - v)
#   tau = kR*(theta_t - theta) + kW*(omega_t - omega)
#
# Dynamics:
#   x_dot = v
#   v_dot = F/m
#   theta_dot = omega
#   omega_dot = tau/I

# ---- Simulation settings ----
dt = 2e-4                 # small dt to mimic high-rate inner loop (e.g., PyElastica substep)
T_sine = 1.5              # seconds of sine tracking
T_step = 1.0              # seconds of step response
T = T_sine + T_step
t = np.arange(0.0, T + dt, dt)

# "Plant" parameters (effective rod tip mass/inertia)
m = 0.02                  # kg (effective translational mass)
I = 5e-5                  # kg*m^2 (effective rotational inertia around some axis)

# Choose gains with near-critical damping:
# For translation: omega_n = sqrt(kp/m), critical kd = 2*sqrt(kp*m)
omega_n_x = 300.0         # rad/s (tracking bandwidth)
kp = m * omega_n_x**2
kd = 2.0 * m * omega_n_x  # critical damping

# For rotation: omega_n = sqrt(kR/I), critical kW = 2*sqrt(kR*I)
omega_n_th = 160.0        # rad/s
kR = I * omega_n_th**2
kW = 2.0 * I * omega_n_th # critical damping

# Optional saturation (helps avoid spikes on abrupt steps)
F_max = 15.0              # N
tau_max = 0.5             # N*m

# ---- Target trajectories (sine then step) ----
# Translation target: sine wave then a step
A_x = 0.04                # meters
f_x = 1.5                 # Hz

x_t = np.zeros_like(t)
v_t = np.zeros_like(t)

# Rotation target: sine wave then a step (in radians)
A_th = np.deg2rad(18.0)   # radians
f_th = 1.2                # Hz

th_t = np.zeros_like(t)
om_t = np.zeros_like(t)

for i, ti in enumerate(t):
    if ti <= T_sine:
        # Sine tracking segment
        x_t[i] = A_x * np.sin(2*np.pi*f_x*ti)
        v_t[i] = A_x * (2*np.pi*f_x) * np.cos(2*np.pi*f_x*ti)

        th_t[i] = A_th * np.sin(2*np.pi*f_th*ti)
        om_t[i] = A_th * (2*np.pi*f_th) * np.cos(2*np.pi*f_th*ti)
    else:
        # Step segment (position step + orientation step)
        x_t[i] = 0.03      # 3 cm step
        v_t[i] = 0.0

        th_t[i] = np.deg2rad(12.0)  # 12 degree step
        om_t[i] = 0.0

# ---- State initialization ----
x = 0.0
v = 0.0
th = 0.0
om = 0.0

x_hist = np.zeros_like(t)
v_hist = np.zeros_like(t)
F_hist = np.zeros_like(t)

th_hist = np.zeros_like(t)
om_hist = np.zeros_like(t)
tau_hist = np.zeros_like(t)

# ---- Time stepping (simple explicit Euler for clarity) ----
# (This is NOT a recommendation for PyElastica integration; it's a minimal standalone test.)
for k in range(len(t)):
    # Impedance forces/torques
    F = kp*(x_t[k] - x) + kd*(v_t[k] - v)
    tau = kR*(th_t[k] - th) + kW*(om_t[k] - om)

    # Saturation to avoid unrealistic spikes on discontinuities
    if abs(F) > F_max:
        F = np.sign(F) * F_max
    if abs(tau) > tau_max:
        tau = np.sign(tau) * tau_max

    # Save
    x_hist[k] = x
    v_hist[k] = v
    F_hist[k] = F

    th_hist[k] = th
    om_hist[k] = om
    tau_hist[k] = tau

    # Dynamics update
    a = F / m
    alpha = tau / I

    # Euler integrate
    v = v + dt * a
    x = x + dt * v

    om = om + dt * alpha
    th = th + dt * om

# ---- Plot 1: Linear tracking ----
plt.figure()
plt.plot(t, x_t, label="x_target")
plt.plot(t, x_hist, label="x")
plt.xlabel("time (s)")
plt.ylabel("position (m)")
plt.title("Impedance coupling — translation tracking (sine then step)")
plt.legend()
plt.grid(True)

# ---- Plot 2: Rotational tracking ----
plt.figure()
plt.plot(t, np.rad2deg(th_t), label="theta_target (deg)")
plt.plot(t, np.rad2deg(th_hist), label="theta (deg)")
plt.xlabel("time (s)")
plt.ylabel("angle (deg)")
plt.title("Impedance coupling — rotation tracking (sine then step)")
plt.legend()
plt.grid(True)

plt.show()
