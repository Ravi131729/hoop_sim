import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --------------------
# Parameters
# --------------------
mo = 0.25
R = 0.07
mp = 0.135
lp = 0.05
g = 9.81
Io = mo * R**2
e = 0.8   # restitution

def torque(t):
    if t < 3:
        tau = 0
    elif t < 3.3:
        tau = 0
    else:
        tau = 0


    return tau
# --------------------
# Rolling dynamics
# --------------------
def rolling_dynamics(t, z):
    x, y, phi, th, dx, dy, dphi, dth = z
    tau = torque(t)

    M = np.zeros((2, 2))
    M[0, 0] = Io + (mo + mp) * R**2
    M[0, 1] = -mp * lp * R * np.cos(th)
    M[1, 0] = M[0, 1]
    M[1, 1] = mp * lp**2

    F = np.zeros((2, 1))
    F[0, 0] = -tau - mp * lp * R * dth**2 * np.sin(th)
    F[1, 0] = tau - mp * lp * g * np.sin(th)

    ddphi, ddth = np.linalg.solve(M, F).flatten()
    ddx = -R * ddphi
    ddy = 0.0

    return [dx, dy, dphi, dth, ddx, ddy, ddphi, ddth]

# --------------------
# Flight dynamics
# --------------------
def flight_dynamics(t, z):
    x, y, phi, th, dx, dy, dphi, dth = z
    tau = 0.011  # torque(t)

    # Re-ordered mass matrix for [ẍ, ÿ, φ̈, θ̈]
    M = np.array([
        [mo + mp, 0, 0,-lp*mp*np.cos(th)],
        [0, mo + mp,0, mp*lp*np.sin(th)],
        [0, 0, Io, 0],
        [-lp*mp*np.cos(th), mp*lp*np.sin(th), Io, 0],
    ])

    # Force vector in same order
    F = np.array([
        -mp*lp*(dth**2)*np.sin(th),
        -(mo+mp)*g - mp*lp*np.cos(th)*(dth**2),
        -tau,
        tau - mp*lp*g*np.sin(th)
    ])

    ddx, ddy, ddphi, ddth = np.linalg.solve(M, F)

    # Match state order [x, y, φ, θ, dx, dy, dφ, dθ]
    return [dx, dy, dphi, dth, ddx, ddy, ddphi, ddth]


# --------------------
# Events
# --------------------
def contact_loss_event(t, z):
    _, _, _, th, _, _, _, dth = z
    lam = (mo + mp) * g + mp * lp * np.cos(th) * dth**2
    return lam
contact_loss_event.terminal = True
contact_loss_event.direction = -1

def impact_event(t, z):
    y, dy = z[1], z[5]
    return y - R


impact_event.terminal = True
impact_event.direction = -1

# --------------------
# Hybrid simulation loop
# --------------------
# t0, tf = 0,1
# # Initial conditions: [phi, theta, x, y, dphi, dtheta, dx, dy]

# z0 = [0, R, 0, -1.57, 1, 5, 0.1, -0.1]  # initial state


# t_current, z_current = t0, z0
# mode = "flight"


# sol = solve_ivp(flight_dynamics, [t_current, tf], z_current,
#                 max_step=1e-3, rtol=1e-8, atol=1e-8)
# plt.figure(figsize=(6,6))
# plt.plot(sol.y[0], sol.y[1], label="Trajectory (x,y)")
# plt.xlabel("x [m]")
# plt.ylabel("y [m]")
# plt.title("Flight Path")
# plt.legend()
# plt.grid()
# plt.axis("equal")
# plt.show()


################################################

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Example tau function (you can replace this with your own)
def get_tau(t, X, b):
    return 1.0  # constant torque for now

def flight_dynamics(t, X, b):
    phi, theta, x, y, dphi, dtheta, dx, dy = X

    tau = get_tau(t, X, b)

    M = np.array([
        [b["Io"] + b["mp"]*b["Lp"]**2,          0,                      0, 0],
        [0,                          b["mp"]*b["Lp"]**2, -b["Lp"]*b["mp"]*np.cos(theta),  b["mp"]*b["Lp"]*np.sin(theta)],
        [0, -b["Lp"]*b["mp"]*np.cos(theta),     b["mo"]+b["mp"],        0],
        [0,  b["mp"]*b["Lp"]*np.sin(theta),     0,                      b["mo"]+b["mp"]]
    ])

    F = np.array([
        tau,
        -tau + b["mp"]*b["Lp"]*b["g"]*np.sin(theta),
        b["mp"]*b["Lp"]*(dtheta**2)*np.sin(theta),
        (b["mo"]+b["mp"])*b["g"] + b["mp"]*b["Lp"]*np.cos(theta)*(dtheta**2)
    ])

    accels = -np.linalg.solve(M, F)  # 4x1 vector

    dXdt = np.array([dphi, dtheta, dx, dy, *accels])
    return dXdt

# System parameters (dictionary instead of struct)
b = {
    "Io": Io,   # inertia
    "mp": mp,   # pendulum mass
    "mo": mo,   # body mass
    "Lp": lp,   # pendulum length
    "g": g    # gravity
}

# Initial conditions: [phi, theta, x, y, dphi, dtheta, dx, dy]
X0 = [0.0, 1.57, 0.0, 0.1, 10, 100, 0.50, 0]

t_span = (0, 1)       # simulate 5 seconds
t_eval = np.linspace(t_span[0], t_span[1], 500)

sol = solve_ivp(flight_dynamics, t_span, X0, args=(b,), t_eval=t_eval, rtol=1e-8, atol=1e-8 , method="Radau")
plt.figure(figsize=(6,6))
plt.plot(sol.y[2], sol.y[3], label="Trajectory (x,y)")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Flight Path")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()
