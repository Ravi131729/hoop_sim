import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --------------------------------------------------
# Problem dimensions
# --------------------------------------------------

nx = 8
nu = 1

N = 500
T = 0.5
dt = 0.001
N = int(T/dt)

# --------------------------------------------------
# Parameters
# --------------------------------------------------

mp = 0.9
mo = 0.6
R = 0.12
Io = mo * R**2
Lp = 0.05
g = 9.81

p_val = np.array([Io, mp, mo, Lp, g])
MAX_SPEED = 300  # Maximum relative speed for the motor/joint
# --------------------------------------------------
# Dynamics
# --------------------------------------------------

Xsym = ca.SX.sym("X", nx)
Usym = ca.SX.sym("U", nu)
Psym = ca.SX.sym("P", 5)

x       = Xsym[0]
y       = Xsym[1]
phi     = Xsym[2]
theta   = Xsym[3]
dx      = Xsym[4]
dy      = Xsym[5]
dphi    = Xsym[6]
dtheta  = Xsym[7]

Io_s = Psym[0]
mp_s = Psym[1]
mo_s = Psym[2]
Lp_s = Psym[3]
g_s  = Psym[4]

tau = Usym[0]

M = ca.vertcat(
    ca.horzcat(mo_s + mp_s, 0, 0, -Lp_s * mp_s * ca.cos(theta)),
    ca.horzcat(0, mo_s + mp_s, 0, Lp_s * mp_s * ca.sin(theta)),
    ca.horzcat(0, 0, Io_s, 0),
    ca.horzcat(-Lp_s * mp_s * ca.cos(theta),
                Lp_s * mp_s * ca.sin(theta),
                0,
                Lp_s**2 * mp_s)
)

F = ca.vertcat(
    -mp_s * Lp_s * ca.sin(theta) * dtheta**2,
    -mp_s * Lp_s * dtheta**2 * ca.cos(theta)
        - (mo_s + mp_s) * g_s,
    -tau,
    tau - mp_s * Lp_s * g_s * ca.sin(theta),
)

qdd = ca.solve(M, F)

xdot = ca.vertcat(
    dx,
    dy,
    dphi,
    dtheta,
    qdd
)

f = ca.Function(
    "f",
    [Xsym, Usym, Psym],
    [xdot]
)
M_fun = ca.Function(
    "M_fun",
    [Xsym, Psym],
    [M]
)
def rk4_step(x, u, p, dt):

    # Relative motor/joint velocity
    omega_rel = x[7] - x[6]

    # Turn torque off at the velocity limit
    if abs(float(omega_rel)) >= MAX_SPEED:
        u_eff = np.array([0.0])
    else:
        u_eff = u

    k1 = f(x, u_eff, p)
    k2 = f(x + dt/2 * k1, u_eff, p)
    k3 = f(x + dt/2 * k2, u_eff, p)
    k4 = f(x + dt * k3, u_eff, p)

    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


data = np.load("optimal_trajectory.npz")

X1_sol = data["X"]
U1_sol = data["U"]
dt1_sol = float(data["dt"])

print("Original X shape:", X1_sol.shape)
print("Original U shape:", U1_sol.shape)


# --------------------------------------------------
# Convert to:
#
# X1_sol : (nx, N+1)
# U1_sol : (nu, N+1)
# --------------------------------------------------

if X1_sol.shape[1] == nx:
    X1_sol = X1_sol.T

if U1_sol.ndim == 1:
    U1_sol = U1_sol.reshape(1, -1)


N_lqr = X1_sol.shape[1] - 1

print("\nAfter reshaping:")
print("X shape:", X1_sol.shape)
print("U shape:", U1_sol.shape)
print("N_lqr:", N_lqr)
print("dt:", dt1_sol)
print("Trajectory duration:", N_lqr * dt1_sol)






# --------------------------------------------------
# 1. Build discrete RK4 dynamics symbolically
# --------------------------------------------------

X_lqr = ca.SX.sym("X_lqr", nx)
U_lqr = ca.SX.sym("U_lqr", nu)

h = float(dt1_sol)

k1 = f(X_lqr, U_lqr, p_val)

k2 = f(
    X_lqr + h/2 * k1,
    U_lqr,
    p_val
)

k3 = f(
    X_lqr + h/2 * k2,
    U_lqr,
    p_val
)

k4 = f(
    X_lqr + h * k3,
    U_lqr,
    p_val
)

X_next = (
    X_lqr
    + h/6 * (k1 + 2*k2 + 2*k3 + k4)
)


# --------------------------------------------------
# 2. Linearize discrete dynamics
#
# delta_x[k+1] = A[k] delta_x[k]
#                + B[k] delta_u[k]
# --------------------------------------------------

A_expr = ca.jacobian(X_next, X_lqr)
B_expr = ca.jacobian(X_next, U_lqr)

AB_fun = ca.Function(
    "AB_fun",
    [X_lqr, U_lqr],
    [A_expr, B_expr]
)


# --------------------------------------------------
# 3. Evaluate A, B along nominal trajectory
# --------------------------------------------------

A_list = []
B_list = []

for k in range(N_lqr):

    # Handle either U shape (1,N) or (1,N+1)
    uk = U1_sol[:, min(k, U1_sol.shape[1] - 1)]

    Ak, Bk = AB_fun(
        X1_sol[:, k],
        uk
    )

    A_list.append(
        np.array(Ak, dtype=float)
    )

    B_list.append(
        np.array(Bk, dtype=float)
    )


# --------------------------------------------------
# 4. TVLQR COST
#
# J = e_N' Qf e_N
#     + sum(e_k' Q e_k + du_k' R du_k)
# --------------------------------------------------

# Start with normalized state-error scales.
#
# Interpretation:
# These are roughly the tracking errors at which
# each state contributes O(1) to the cost.

state_scale = np.array([
    0.10,     # x       [m]
    0.02,     # y       [m]
    0.20,     # phi     [rad]
    0.10,     # theta   [rad]
    1.00,     # dx      [m/s]
    1.00,     # dy      [m/s]
    10.0,     # dphi    [rad/s]
    10.0      # dtheta  [rad/s]
])

Q = np.diag(
    1.0 / state_scale**2
)

# Cost on deviation from nominal torque
R_lqr = np.array([[2.0]])

# We specifically care about reaching the
# nominal terminal pre-impact state.
Qf = 30.0 * Q


# --------------------------------------------------
# 5. Backward Riccati recursion
# --------------------------------------------------

K_list = [None] * N_lqr
P_list = [None] * (N_lqr + 1)

P = Qf.copy()

P_list[N_lqr] = P.copy()

for k in reversed(range(N_lqr)):

    A = A_list[k]
    B = B_list[k]

    S = (
        R_lqr
        + B.T @ P @ B
    )

    # K = (R + B'PB)^-1 B'PA
    K = np.linalg.solve(
        S,
        B.T @ P @ A
    )

    # Riccati equation
    P = (
        Q
        + A.T @ P @ A
        - A.T @ P @ B @ K
    )

    # Remove tiny numerical asymmetry
    P = 0.5 * (P + P.T)

    K_list[k] = K
    P_list[k] = P.copy()


K_array = np.array(K_list)
P_array = np.array(P_list)

print("K shape:", K_array.shape)
print("P shape:", P_array.shape)

print("\nK[0] =")
print(K_array[0])

print("\nK[last] =")
print(K_array[-1])


# ==================================================
# NONLINEAR TVLQR TEST
# ==================================================

x0_nom = X1_sol[:, 0].copy()

x0_test = x0_nom.copy()

# Example perturbations
x0_test[3] += 0*0.2     # theta +0.02 rad
x0_test[7] += 10.0      # dtheta +1 rad/s


X_tvlqr = np.zeros(
    (nx, N_lqr + 1)
)

U_tvlqr = np.zeros(
    (nu, N_lqr)
)

X_tvlqr[:, 0] = x0_test


# Use the same torque limit as optimization
u_max = 1.0


for k in range(N_lqr):

    x_current = X_tvlqr[:, k]

    x_ref = X1_sol[:, k]

    u_ref = U1_sol[
        :,
        min(k, U1_sol.shape[1] - 1)
    ]

    K = K_list[k]

    # ----------------------------------------------
    # TVLQR
    #
    # u = u* - K(x-x*)
    # ----------------------------------------------

    error = x_current - x_ref

    u_fb = -K @ error

    u_cmd = u_ref + u_fb

    # Torque saturation
    u_cmd = np.clip(
        u_cmd,
        -u_max,
        u_max
    )

    # ----------------------------------------------
    # Motor relative-speed limit
    # ----------------------------------------------

    omega_rel = (
        x_current[7]
        - x_current[6]
    )

    if abs(omega_rel) >= MAX_SPEED:

        u_cmd = np.array([0.0])


    U_tvlqr[:, k] = u_cmd


    # ----------------------------------------------
    # NONLINEAR plant simulation
    # ----------------------------------------------

    x_next = rk4_step(
        x_current,
        u_cmd,
        p_val,
        h
    )

    X_tvlqr[:, k + 1] = (
        np.array(x_next).flatten()
    )


    # --------------------------------------------------
# Terminal tracking error
# --------------------------------------------------

terminal_error = (
    X_tvlqr[:, -1]
    - X1_sol[:, -1]
)

normalized_terminal_error = (
    terminal_error
    / state_scale
)

print("\n==============================")
print("TVLQR TERMINAL RESULTS")
print("==============================")

print("\nTerminal state nominal:")
print(X1_sol[:, -1])

print("\nTerminal state TVLQR:")
print(X_tvlqr[:, -1])

print("\nTerminal error:")
print(terminal_error)

print(
    "\nNormalized terminal error norm =",
    np.linalg.norm(normalized_terminal_error)
)

time = np.arange(N_lqr + 1) * h

plt.figure(figsize=(9, 10))


# --------------------------------------------------
# y
# --------------------------------------------------

plt.subplot(4, 1, 1)

plt.plot(
    time,
    X1_sol[1, :],
    label="Nominal",
    linewidth=2
)

plt.plot(
    time,
    X_tvlqr[1, :],
    "--",
    label="TVLQR",
    linewidth=2
)

plt.ylabel("y [m]")
plt.legend()
plt.grid(True)


# --------------------------------------------------
# theta
# --------------------------------------------------

plt.subplot(4, 1, 2)

plt.plot(
    time,
    X1_sol[3, :],
    label="Nominal",
    linewidth=2
)

plt.plot(
    time,
    X_tvlqr[3, :],
    "--",
    label="TVLQR",
    linewidth=2
)

plt.ylabel("theta [rad]")
plt.legend()
plt.grid(True)


# --------------------------------------------------
# dtheta
# --------------------------------------------------

plt.subplot(4, 1, 3)

plt.plot(
    time,
    X1_sol[7, :],
    label="Nominal",
    linewidth=2
)

plt.plot(
    time,
    X_tvlqr[7, :],
    "--",
    label="TVLQR",
    linewidth=2
)

plt.ylabel("dtheta [rad/s]")
plt.legend()
plt.grid(True)


# --------------------------------------------------
# Torque
# --------------------------------------------------

plt.subplot(4, 1, 4)

plt.plot(
    time[:-1],
    U1_sol[0, :N_lqr],
    label="Nominal torque",
    linewidth=2
)

plt.plot(
    time[:-1],
    U_tvlqr[0, :],
    "--",
    label="TVLQR torque",
    linewidth=2
)

plt.axhline(
    u_max,
    linestyle=":"
)

plt.axhline(
    -u_max,
    linestyle=":"
)

plt.xlabel("Time [s]")
plt.ylabel("Torque [Nm]")
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(
    X1_sol[0, :],
    X1_sol[1, :],
    label="Nominal",
    linewidth=2
)
plt.plot(
    X_tvlqr[0, :],
    X_tvlqr[1, :],
    "--",
    label="TVLQR",
    linewidth=2
)
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.legend()
plt.grid(True)
plt.show()