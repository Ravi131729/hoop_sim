import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Problem dimensions
# --------------------------------------------------

nx = 8
nu = 1

N =15    # total nodes, fixed


dt_nominal = 0.01   # only used to seed the initial guess
dt_min = 1e-2
dt_max = 0.05

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

e =0.5      # coefficient of restitution
mu = 0.3       # coefficient of friction
y_contact = 0.100


u_max = 1.0
max_speed = 200.0   # rad/s, bound on relative joint speed dtheta - dphi

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

# --------------------------------------------------
# Initial condition
# --------------------------------------------------

x0 = np.array([0.0, 0.3, 0.0, 0.0,
               0.5, 0.0, 5.0, 0.0])

u_val = np.array([-0.0])

# --------------------------------------------------
# Numeric impact map (mirrors test.py) -- used only to
# pick a nominal pre-impact state and fix the contact
# mode (stick/slip) ahead of building the NLP.
# --------------------------------------------------


def rk4_step_num(xk, uk, p, dt):
    k1 = f(xk, uk, p)
    k2 = f(xk + dt / 2 * k1, uk, p)
    k3 = f(xk + dt / 2 * k2, uk, p)
    k4 = f(xk + dt * k3, uk, p)
    return np.array(xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)).flatten()


def impact_map_num(qdot_minus, q, p, e, mu, R):
    M_num = np.array(M_fun(np.concatenate([q, qdot_minus]), p)).astype(float)
    Minv = np.linalg.inv(M_num)

    a_t = np.array([1.0, 0.0, -R, 0.0])
    a_n = np.array([0.0, 1.0, 0.0, 0.0])

    v_t_minus = qdot_minus[0] - R * qdot_minus[2]
    v_n_minus = qdot_minus[1]

    Ktt = a_t @ Minv @ a_t
    Ktn = a_t @ Minv @ a_n
    Knt = a_n @ Minv @ a_t
    Knn = a_n @ Minv @ a_n

    A = np.array([[Ktt, Ktn], [Knt, Knn]])
    b = np.array([-v_t_minus, -(1.0 + e) * v_n_minus])
    lambda_t_stick, lambda_n_stick = np.linalg.solve(A, b)

    if lambda_n_stick >= 0 and abs(lambda_t_stick) <= mu * lambda_n_stick:
        mode = "stick"
        lambda_t, lambda_n = lambda_t_stick, lambda_n_stick
        sign_vt = 0.0
    else:
        mode = "slip"
        sign_vt = np.sign(v_t_minus)
        denom = Knn - mu * sign_vt * Knt
        lambda_n = max(-(1.0 + e) * v_n_minus / denom, 0.0)
        lambda_t = -mu * lambda_n * sign_vt

    impulse = a_t * lambda_t + a_n * lambda_n
    qdot_plus = qdot_minus + Minv @ impulse

    return qdot_plus, lambda_t, lambda_n, mode, sign_vt





# --------------------------------------------------
# Symbolic impact map, with the stick/slip branch fixed
# to `contact_mode` so it is smooth inside the NLP.
# --------------------------------------------------

q_imp = ca.SX.sym("q_imp", 4)
qdotm_imp = ca.SX.sym("qdotm_imp", 4)
p_imp = ca.SX.sym("p_imp", 5)

M_imp = M_fun(ca.vertcat(q_imp, qdotm_imp), p_imp)
Minv_imp = ca.inv(M_imp)

a_t = ca.SX(np.array([1.0, 0.0, -R, 0.0]))
a_n = ca.SX(np.array([0.0, 1.0, 0.0, 0.0]))

v_t_minus_imp = qdotm_imp[0] - R * qdotm_imp[2]
v_n_minus_imp = qdotm_imp[1]

Ktt_imp = a_t.T @ Minv_imp @ a_t
Ktn_imp = a_t.T @ Minv_imp @ a_n
Knt_imp = a_n.T @ Minv_imp @ a_t
Knn_imp = a_n.T @ Minv_imp @ a_n
contact_mode = "stick"  # or "slip"
if contact_mode == "stick":
    A_imp = ca.vertcat(
        ca.horzcat(Ktt_imp, Ktn_imp),
        ca.horzcat(Knt_imp, Knn_imp)
    )
    b_imp = ca.vertcat(-v_t_minus_imp, -(1.0 + e) * v_n_minus_imp)
    lam_imp = ca.solve(A_imp, b_imp)
    lambda_t_imp = lam_imp[0]
    lambda_n_imp = lam_imp[1]
else:
    denom_imp = Knn_imp - mu * contact_sign_vt * Knt_imp
    lambda_n_imp = -(1.0 + e) * v_n_minus_imp / denom_imp
    lambda_t_imp = -mu * lambda_n_imp * contact_sign_vt

impulse_imp = a_t * lambda_t_imp + a_n * lambda_n_imp
qdot_plus_imp = qdotm_imp + Minv_imp @ impulse_imp

impact_map_fun = ca.Function(
    "impact_map_fun",
    [q_imp, qdotm_imp, p_imp],
    [qdot_plus_imp]
)
#com velocity after impact
# qdot_plus_com = qdot_plus_imp[0:4]

X_com = ca.SX.sym("X_com", nx)

dx_com     = X_com[4]
dy_com     = X_com[5]
theta_com  = X_com[3]
dtheta_com = X_com[7]

vx_com = (
    (mo + mp) * dx_com
    - mp * Lp * ca.cos(theta_com) * dtheta_com
) / (mo + mp)

vy_com = (
    (mo + mp) * dy_com
    + mp * Lp * ca.sin(theta_com) * dtheta_com
) / (mo + mp)

com_velocity_fun = ca.Function(
    "com_velocity_fun",
    [X_com],
    [ca.vertcat(vx_com, vy_com)]
)


# --------------------------------------------------
# Hermite-Simpson collocation defect
# --------------------------------------------------


def hermite_simpson_defect(xk, xk1, uk, uk1, dt, p):
    fk = f(xk, uk, p)
    fk1 = f(xk1, uk1, p)
    u_mid = 0.5 * (uk + uk1)
    x_mid = 0.5 * (xk + xk1) + dt / 8 * (fk - fk1)
    f_mid = f(x_mid, u_mid, p)
    return xk1 - xk - dt / 6 * (fk + 4 * f_mid + fk1)


# --------------------------------------------------
# Build the two-phase collocation NLP
# --------------------------------------------------

opti = ca.Opti()

X1 = opti.variable(nx, N + 1)
U1 = opti.variable(nu, N + 1)


dt1 = opti.variable()      # phase-1 time step (before contact), free

opti.subject_to(opti.bounded(dt_min, dt1, dt_max))

# Initial condition
opti.subject_to(X1[:, 0] == x0)

# Phase 1: collocation defects, ground clearance, control bounds
for k in range(N):
    opti.subject_to(
        hermite_simpson_defect(
            X1[:, k], X1[:, k + 1], U1[:, k], U1[:, k + 1], dt1, p_val
        ) == 0
    )

#max speed constraint on relative joint speed dtheta - dphi
opti.subject_to(opti.bounded(-max_speed, X1[7,:] - X1[6,:], max_speed))

opti.subject_to(opti.bounded(-u_max, U1, u_max))

# Fixed contact node: body touches the ground exactly at node N1
opti.subject_to(X1[1, N] == y_contact)
# opti.subject_to(X1[5, N] <= 0.0)  # downward velocity at contact
# opti.subject_to(X1[1, 0:N-1] > y_contact)  # zero angular position at contact
q_contact = X1[0:4, N]
qdot_minus_contact = X1[4:8, N]


qdot_plus = impact_map_fun(q_contact, qdot_minus_contact, p_val)

X_plus_contact = ca.vertcat(q_contact, qdot_plus)

v_com = com_velocity_fun(X_plus_contact)



opti.subject_to(v_com[0] >= 0.0)  # downward velocity of COM after impact
opti.subject_to(v_com[1] ==1.80)  # forward velocity of COM after impact
# opti.subject_to(X1[7,N] ==0.00)

# opti.subject_to(X1[3,N] == 2.0)




# Actual objective
J = 0
for k in range(N):
    J += dt1/2 * (
        ca.sumsqr(U1[:,k]) +
        ca.sumsqr(U1[:,k+1])
    )

opti.minimize(J)

# --------------------------------------------------
# Initial guess: rollout with u = 0.5, dt = 0.1 s
# --------------------------------------------------

dt1_guess = 0.01
u_guess = np.array([0.7])

X1_guess = np.zeros((nx, N + 1))
X1_guess[:, 0] = x0

for k in range(N):
    X1_guess[:, k + 1] = rk4_step_num(
        X1_guess[:, k],
        u_guess,
        p_val,
        dt1_guess
    )

# Set initial guesses
opti.set_initial(X1, X1_guess)
opti.set_initial(U1, u_guess)
opti.set_initial(dt1, dt1_guess)





# opti.set_initial(X1, X1_guess.T)
# opti.set_initial(U1, 0.0)
# opti.set_initial(dt1, dt1_guess)


opti.solver("ipopt", {}, {
    "max_iter": 3000,
    "acceptable_tol": 1e-5,
    "acceptable_constr_viol_tol": 1e-5,
    "acceptable_iter": 10,
})

sol = opti.solve()

# --------------------------------------------------
# Extract phase-1 solution
# --------------------------------------------------

X1_sol = sol.value(X1)                      # nx x (N+1)
U1_sol = sol.value(U1).reshape(nu, -1)     # nu x (N+1)
dt1_sol = sol.value(dt1)

# Time vector
time = np.linspace(0, N * dt1_sol, N + 1)

T1 = N * dt1_sol

print(f"dt1 = {dt1_sol:.6f} s")
print(f"Phase-1 duration T1 = {T1:.4f} s")
print(f"Terminal height y(T1) = {X1_sol[1, -1]:.6f} m")
print(f"Terminal theta = {X1_sol[3, -1]:.6f} rad")
print(f"Terminal theta_dot = {X1_sol[7, -1]:.6f} rad/s")

# --------------------------------------------------
# Terminal optimization state = PRE-IMPACT state
# --------------------------------------------------

x_minus = X1_sol[:, -1].copy()

q_minus = x_minus[0:4]
qdot_minus = x_minus[4:8]

print("\nPre-impact terminal state:")
print(x_minus)


# --------------------------------------------------
# Apply impact map
# --------------------------------------------------

qdot_plus, lambda_t, lambda_n, mode, sign_vt = impact_map_num(
    qdot_minus,
    q_minus,
    p_val,
    e,
    mu,
    R
)


# --------------------------------------------------
# Construct POST-IMPACT state
# Position is unchanged, velocity jumps
# --------------------------------------------------

x_plus = x_minus.copy()
x_plus[4:8] = qdot_plus


# --------------------------------------------------
# COM velocity before and after impact
# --------------------------------------------------

v_com_minus = np.array(
    com_velocity_fun(x_minus)
).flatten()

v_com_plus = np.array(
    com_velocity_fun(x_plus)
).flatten()


print("\n========== IMPACT ==========")

print("Impact mode:", mode)
print(f"lambda_t = {lambda_t:.6f}")
print(f"lambda_n = {lambda_n:.6f}")

print("\nGeneralized velocity:")
print("qdot_minus =", qdot_minus)
print("qdot_plus  =", qdot_plus)

print("\nCOM velocity PRE impact:")
print(f"vx_com- = {v_com_minus[0]:.6f} m/s")
print(f"vy_com- = {v_com_minus[1]:.6f} m/s")

print("\nCOM velocity POST impact:")
print(f"vx_com+ = {v_com_plus[0]:.6f} m/s")
print(f"vy_com+ = {v_com_plus[1]:.6f} m/s")

print("\nCOM velocity change:")
print(f"delta vx_com = {v_com_plus[0] - v_com_minus[0]:.6f} m/s")
print(f"delta vy_com = {v_com_plus[1] - v_com_minus[1]:.6f} m/s")


# --------------------------------------------------
# Plots
# --------------------------------------------------

plt.figure(figsize=(7, 8))

# Height
plt.subplot(3, 1, 1)
plt.plot(time, X1_sol[1, :], linewidth=2)
plt.axhline(
    y_contact,
    color="r",
    linestyle="--",
    label="y_contact"
)
plt.xlabel("Time [s]")
plt.ylabel("y [m]")
plt.title("Height vs time")
plt.legend()
plt.grid(True)


# theta_dot
plt.subplot(3, 1, 2)
plt.plot(time, X1_sol[7, :], linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel(r"$\dot{\theta}$ [rad/s]")
plt.title("Angular velocity")
plt.grid(True)


# Control torque
plt.subplot(3, 1, 3)
plt.plot(time, U1_sol[0, :], linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel(r"$\tau$ [N m]")
plt.title("Control input")
plt.grid(True)


plt.tight_layout()
plt.show()

# --------------------------------------------------
# Forward RK4 simulation of optimized control
# Including impacts -- for plotting/verification
# --------------------------------------------------

substeps = 50
dt_sim = dt1_sol / substeps

T = N * dt1_sol

Xsim = [x0.copy()]
time_sim = [0.0]

x_sim = x0.copy()
count = 0

impact_times = []

for k in range(N+30):

    # Optimized controls at the two collocation nodes
    if k >= N:
        u_k = np.array([0.0])
        u_k1 = np.array([0.0])
    else:
        u_k  = U1_sol[:, k]
        u_k1 = U1_sol[:, k + 1]

    for j in range(substeps):

        # Linear interpolation of optimized control
        alpha_u = j / substeps
        u_sim = (1.0 - alpha_u) * u_k + alpha_u * u_k1

        x_old = x_sim.copy()

        # ------------------------------------------
        # RK4 integration
        # ------------------------------------------
        x_new = rk4_step_num(
            x_old,
            u_sim,
            p_val,
            dt_sim
        )

        t_old = time_sim[-1]
        t_new = t_old + dt_sim

        # ------------------------------------------
        # Collision detection
        # ------------------------------------------
        if x_old[1] > y_contact and x_new[1] <= y_contact:

            print(f"Impact detected near t = {t_new:.6f} s")

            # Estimate where inside the integration
            # step contact occurred
            alpha_contact = (
                (x_old[1] - y_contact)
                / (x_old[1] - x_new[1])
            )

            # Interpolate state to contact
            x_contact = (
                x_old
                + alpha_contact * (x_new - x_old)
            )

            x_contact[1] = y_contact

            t_contact = (
                t_old
                + alpha_contact * dt_sim
            )

            # --------------------------------------
            # Pre-impact velocity
            # --------------------------------------
            qdot_minus = x_contact[4:8].copy()

            # --------------------------------------
            # Apply impact map
            # --------------------------------------
            qdot_plus, lambda_t, lambda_n, mode, sign_vt = (
                impact_map_num(
                    qdot_minus,
                    x_contact[0:4],
                    p_val,
                    e,
                    mu,
                    R
                )
            )

            print("qdot_minus =", qdot_minus)
            print("qdot_plus  =", qdot_plus)
            print("Impact mode =", mode)
            print("lambda_t =", lambda_t)
            print("lambda_n =", lambda_n)

            # Post-impact state
            x_contact[4:8] = qdot_plus

            x_new = x_contact.copy()

            count += 1
            impact_times.append(t_contact)

            print(f"Number of impacts: {count}")

        # ------------------------------------------
        # Save simulation
        # ------------------------------------------
        x_sim = x_new.copy()

        Xsim.append(x_sim.copy())
        time_sim.append(t_new)


Xsim = np.array(Xsim)
time_sim = np.array(time_sim)


# --------------------------------------------------
# Collocation node times
# --------------------------------------------------

time_opt = np.linspace(
    0,
    N * dt1_sol,
    N + 1
)


# --------------------------------------------------
# Plots
# --------------------------------------------------
plt.figure(figsize=(8, 10))
plt.plot(
    X1_sol[0, :],
    X1_sol[1, :],
    "o-",
    label="Optimization"
)
# plt.plot(
#     time_opt,
#     X1_sol[0, :],
#     "--",
#     label="Optimization"
# )

plt.plot(
    Xsim[:, 0],
    Xsim[:, 1],
    "--",
    label="RK4 + impact"
)

plt.axhline(
    y_contact,
    color="r",
    linestyle="--",
    label="ground"
)

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("x-y trajectory")
plt.legend()
plt.grid(True)




plt.figure(figsize=(8, 10))


# phi dot
plt.subplot(4, 1, 1)

plt.plot(
    time_opt,
    X1_sol[6, :],
    "o-",
    label="Optimization"

)

plt.plot(
    time_sim,
    Xsim[:, 6],
    "--",
    label="RK4 + impact"
)

plt.ylabel(r"$\dot{\phi}$ [rad/s]")
plt.legend()
plt.grid(True)

def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# theta
plt.subplot(4, 1, 2)

plt.plot(
    time_opt,
    wrap_angle(X1_sol[3, :]),
    "o-",
    label="Optimization"
)

plt.plot(
    time_sim,
    wrap_angle(Xsim[:, 3]),
    "--",
    label="RK4 + impact"
)

plt.ylabel(r"$\theta$ [rad]")
plt.legend()
plt.grid(True)


# theta_dot
plt.subplot(4, 1, 3)

plt.plot(
    time_opt,
    X1_sol[7, :],
    "o-",
    label="Optimization"
)

plt.plot(
    time_sim,
    Xsim[:, 7],
    "--",
    label="RK4 + impact"
)

plt.ylabel(r"$\dot{\theta}$ [rad/s]")
plt.legend()
plt.grid(True)


# Optimized torque
plt.subplot(4, 1, 4)

plt.plot(
    time_opt,
    U1_sol[0, :],
    "o-"
)

plt.plot(
    time_sim,
    np.interp(time_sim, time_opt, U1_sol[0, :]),
    "--"
)

plt.xlabel("Time [s]")
plt.ylabel(r"$\tau$ [Nm]")
plt.grid(True)

plt.tight_layout()
plt.show()
#relative angular velocity dtheta - dphi
plt.figure(figsize=(6, 4))
plt.plot(
    time_opt,
    X1_sol[7, :] - X1_sol[6, :],
    "o-",
    label="Optimization"
)
plt.xlabel("Time [s]")
plt.ylabel(r"$\dot{\theta} - \dot{\phi}$ [rad/s]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
