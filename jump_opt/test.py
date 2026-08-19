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

mo = 0.5
R = 0.1
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


#angular momentum about COM

def angular_momentum(x, p):
    dx = x[4]
    dy = x[5]
    dphi = x[6]
    dtheta = x[7]

    Io_s = p[0]
    mp_s = p[1]
    mo_s = p[2]
    Lp_s = p[3]

    # Position of the center of mass
    x_com = ((mo_s + mp_s) * x[0] - mp_s * Lp_s * ca.sin(x[3])) / (mo_s + mp_s)
    y_com = ((mo_s + mp_s) * x[1] - mp_s * Lp_s * ca.cos(x[3])) / (mo_s + mp_s)

    # Velocity of the center of mass
    dx_com = ((mo_s + mp_s) * dx - mp_s * Lp_s * ca.cos(x[3]) * dtheta) / (mo_s + mp_s)
    dy_com = ((mo_s + mp_s) * dy + mp_s * Lp_s * ca.sin(x[3]) * dtheta) / (mo_s + mp_s)

    #hoop vel relative to COM
    v_rel_x = dx - dx_com
    v_rel_y = dy - dy_com

    #pend pos
    pend_x = x[0] - Lp_s * ca.sin(x[3])
    pend_y = x[1] - Lp_s * ca.cos(x[3])

    #pend vel
    pend_vel_x = dx - Lp_s * ca.cos(x[3]) * dtheta
    pend_vel_y = dy + Lp_s * ca.sin(x[3]) * dtheta

    #pend vel relative to COM
    v_rel_pend_x = pend_vel_x - dx_com
    v_rel_pend_y = pend_vel_y - dy_com

    r_hoop_COM = ca.vertcat(x[0] - x_com, x[1] - y_com)
    r_pend_COM = ca.vertcat(pend_x - x_com, pend_y - y_com)




    H_hoop_com = (
        Io_s * dphi
        - mo_s * (
            r_hoop_COM[0] * v_rel_y
            - r_hoop_COM[1] * v_rel_x
        )
    )

    H_pend_com = -(
        mp_s * (
            r_pend_COM[0] * v_rel_pend_y
            - r_pend_COM[1] * v_rel_pend_x
        )
    )

    H_com = H_hoop_com + H_pend_com

    return H_com




# --------------------------------------------------
def impact_map(qdot_minus, q, p, e, mu, R):

    # Mass matrix at impact
    M_num = np.array(
        M_fun(
            np.concatenate([q, qdot_minus]),
            p
        )
    ).astype(float)

    Minv = np.linalg.inv(M_num)

    # Contact Jacobian columns associated with impulses
    a_t = np.array([1.0, 0.0, -R, 0.0])
    a_n = np.array([0.0, 1.0, 0.0, 0.0])

    # Pre-impact contact velocities
    v_t_minus = qdot_minus[0] - R*qdot_minus[2]
    v_n_minus = qdot_minus[1]

    # Only apply impact if approaching the contact
    if v_n_minus >= 0:
        return qdot_minus, 0.0, 0.0, "no_impact"

    # --------------------------------------------------
    # Effective inverse mass at contact
    # --------------------------------------------------

    Ktt = a_t @ Minv @ a_t
    Ktn = a_t @ Minv @ a_n
    Knt = a_n @ Minv @ a_t
    Knn = a_n @ Minv @ a_n

    # --------------------------------------------------
    # FIRST: assume sticking
    #
    # vt+ = 0
    # vn+ = -e vn-
    # --------------------------------------------------

    A = np.array([
        [Ktt, Ktn],
        [Knt, Knn]
    ])

    b = np.array([
        -v_t_minus,
        -(1.0 + e)*v_n_minus
    ])

    lambda_t_stick, lambda_n_stick = np.linalg.solve(A, b)

    # --------------------------------------------------
    # Check Coulomb condition
    # --------------------------------------------------

    if lambda_n_stick >= 0 and \
       abs(lambda_t_stick) <= mu * lambda_n_stick:

        # STICKING
        lambda_t = lambda_t_stick
        lambda_n = lambda_n_stick
        mode = "stick"

    else:

        # --------------------------------------------------
        # SLIDING
        # --------------------------------------------------

        sign_vt = np.sign(v_t_minus)

        # lambda_t = -mu*lambda_n*sign(vt-)
        #
        # vn+ = vn- + Knt*lambda_t + Knn*lambda_n
        #
        # solve for lambda_n

        denominator = Knn - mu * sign_vt * Knt

        lambda_n = (
            -(1.0 + e) * v_n_minus
            / denominator
        )

        lambda_n = max(lambda_n, 0.0)

        lambda_t = (
            -mu * lambda_n * sign_vt
        )

        mode = "slip"

    # --------------------------------------------------
    # Apply impulse
    # --------------------------------------------------

    impulse = (
        a_t * lambda_t +
        a_n * lambda_n
    )

    qdot_plus = (
        qdot_minus +
        Minv @ impulse
    )

    return qdot_plus, lambda_t, lambda_n, mode
# --------------------------------------------------
# RK4 discrete dynamics
# --------------------------------------------------

# def rk4_step(x, u, p, dt):
#     k1 = f(x, u, p)
#     k2 = f(x + dt/2 * k1, u, p)
#     k3 = f(x + dt/2 * k2, u, p)
#     k4 = f(x + dt * k3, u, p)

#     return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
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

# --------------------------------------------------
# Initial conditions
# --------------------------------------------------

# State:
# [ x, y, phi, theta,
#  dx, dy,dphi, dtheta]
e =0.5# coefficient of restitution
mu = 0.3     # coefficient of friction

y_contact = 0.1
x0 = np.array([0.0, 0.25, 0.0, 0.0,
               1.0, 0.0, 10.0, 0.0])


u_val = np.array([1.0])

# --------------------------------------------------
# Forward simulation
# --------------------------------------------------

Xsim = np.zeros((N + 1, nx))
Xsim[0, :] = x0

# for k in range(N):
#     Xsim[k + 1, :] = np.array(
#         rk4_step(Xsim[k, :], u_val, p_val, dt)
#     ).flatten()
time = np.linspace(0, T, N + 1)
count = 0
for k in range(N):

    x_old = Xsim[k, :]

    # Integrate one step
    x_new = np.array(
        rk4_step(x_old, u_val, p_val, dt)
    ).flatten()
    if count > 0:
        u_val = np.array([0.0])  # Turn off torque after 0.1 seconds

    # Collision detection:
    # coming from above and crossing y = 0.1




    if x_old[1] > y_contact and x_new[1] <= y_contact:
        print(f"Impact detected at t = {time[k+1]:.4f} s")
        impact_time = time[k+1]
        if count == 0:
          impact_index = k+1

        # Put position exactly at contact
        x_new[1] = y_contact

        x_minus = x_new.copy()

        qdot_minus = x_new[4:8].copy()


        qdot_plus, lambda_t, lambda_n, mode = impact_map(
            qdot_minus,
            x_new[0:4],
            p_val,
            e,
            mu,
            R
        )
        print("qdot_plus =", qdot_plus)

        x_new[1] = y_contact
        x_new[4:8] = qdot_plus

        x_plus = x_new.copy()

        print("Impact mode:", mode)
        print("lambda_t =", lambda_t)
        print("lambda_N =", lambda_n)
        count += 1
        print(f"Number of impacts: {count}")
    Xsim[k + 1, :] = x_new
# Time vector

# --------------------------------------------------
# Extract states
# --------------------------------------------------

phidot_sim   = Xsim[:, 6]
thetadot_sim = Xsim[:, 7]
x_sim     = Xsim[:, 0]
y_sim     = Xsim[:, 1]
phi_sim   = Xsim[:, 2]
theta_sim = Xsim[:, 3]
x_dot_sim = Xsim[:, 4]
y_dot_sim = Xsim[:, 5]

H_com = np.zeros(N + 1)
for k in range(N + 1):
    H_com[k] = float(angular_momentum(Xsim[k, :], p_val))


hoop_vel = np.array([x_dot_sim, y_dot_sim])
hoop_vel_unit = hoop_vel / np.linalg.norm(
    hoop_vel,
    axis=0,
    keepdims=True
)
hoop_x_test =  x_sim[impact_index]+0.05*hoop_vel_unit[0, impact_index]
hoop_y_test =  y_sim[impact_index]+0.05*hoop_vel_unit[1, impact_index]

hoop_x_test_before =  x_sim[impact_index-1]+0.05*hoop_vel_unit[0, impact_index-1]
hoop_y_test_before =  y_sim[impact_index-1]+0.05*hoop_vel_unit[1, impact_index-1]
print("hoop_vel_unit:", hoop_vel[:, impact_index])
#relative velocity
omega_rel_sim = thetadot_sim - phidot_sim
theta_sim = np.mod(theta_sim, 2 * np.pi)
#com position
x_com = ((mo + mp) * x_sim - mp * Lp * np.sin(theta_sim))/(mo + mp)
y_com = ((mo + mp) * y_sim - mp * Lp * np.cos(theta_sim))/(mo + mp)

#pendulum position
pend_x = x_sim - Lp * np.sin(theta_sim)
pend_y = y_sim - Lp * np.cos(theta_sim)

#pend_velocity
pend_x_dot = x_dot_sim - Lp * np.cos(theta_sim) * thetadot_sim
pend_y_dot = y_dot_sim + Lp * np.sin(theta_sim) * thetadot_sim

#com velocity
x_dot_com = ((mo + mp) * x_dot_sim - mp * Lp *np.cos(theta_sim) * thetadot_sim)/(mo + mp)
y_dot_com = ((mo + mp) * y_dot_sim + mp * Lp *np.sin(theta_sim) * thetadot_sim)/(mo + mp)

pend_vel = np.array([pend_x_dot, pend_y_dot])

pend_vel_unit = pend_vel / np.linalg.norm(
    pend_vel,
    axis=0,
    keepdims=True
)


com_vel = np.array([x_dot_com, y_dot_com])

com_vel_unit = com_vel / np.linalg.norm(
    com_vel,
    axis=0,
    keepdims=True
)
pend_x_text = pend_x[impact_index] + 0.05 * pend_vel_unit[0, impact_index]
pend_y_text = pend_y[impact_index] + 0.05 * pend_vel_unit[1, impact_index]

pend_x_text_before = pend_x[impact_index-1] + 0.05 * pend_vel_unit[0, impact_index-1]
pend_y_text_before = pend_y[impact_index-1] + 0.05 *pend_vel_unit[1, impact_index-1]

x_test = x_com[impact_index] + 0.05 * com_vel_unit[0, impact_index]
y_test = y_com[impact_index] + 0.05 * com_vel_unit[1, impact_index]
x_test_before = x_com[impact_index-1] + 0.05 * com_vel_unit[0, impact_index-1]
y_test_before = y_com[impact_index-1] + 0.05 * com_vel_unit[1, impact_index-1]
#

ydot_com_impact = y_dot_com[impact_index]

max_height = ydot_com_impact**2/(2*g)

print("Maximum height after impact:", max_height+R)
# --------------------------------------------------
# Plot x-y trajectory
# --------------------------------------------------

plt.figure(figsize=(7, 6))
plt.plot(x_com, y_com, linewidth=2, label='COM trajectory')
plt.plot(x_sim, y_sim, linewidth=2, label='hoop centertrajectory')
# for xs, ys, xc, yc in zip(x_sim, y_sim, x_com, y_com):
#     plt.plot([xs, xc], [ys, yc], 'k-', linewidth=1)
plt.plot([x_sim[impact_index], x_com[impact_index]], [y_sim[impact_index], y_com[impact_index]], 'k-', linewidth=1, label='Impact')
plt.plot(x_com[impact_index], y_com[impact_index], 'o', markersize=8, label='Impact COM')



plt.plot(x_com[0], y_com[0], 'o', label='Start')
plt.plot(x_com[-1], y_com[-1], 'x', markersize=8, label='End')
circle = Circle(
    (x_sim[impact_index], y_sim[impact_index]),
    0.1,
    fill=False,
    edgecolor='r'
)
plt.gca().add_patch(circle)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Forward Simulation: x-y Trajectory")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


#plot angular momentum
plt.figure(figsize=(7, 6))
plt.plot(time, H_com, linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Angular Momentum about COM [kg.m^2/s]")
plt.title("Forward Simulation: Angular Momentum about COM vs Time")
plt.axvline(impact_time, color='r', linestyle='--', label='Impact Time')
plt.grid(True)
plt.legend()
















plt.figure(figsize=(12,15))
plt.plot([x_sim[impact_index], pend_x[impact_index]], [y_sim[impact_index], pend_y[impact_index]], 'k-', linewidth=1, label='Impact')
plt.plot(x_com[impact_index], y_com[impact_index], 'o', markersize=8, label='Impact COM')
plt.plot(pend_x[impact_index], pend_y[impact_index], 'o', markersize=8, label='Impact Pendulum')
plt.plot(x_sim[impact_index], y_sim[impact_index], 'o', markersize=8, label='Impact Hoop Center')


# plt.plot([x_test, x_com[impact_index]], [y_test, y_com[impact_index]], 'r-', linewidth=1, label='Test Point')
plt.annotate(
    '',
    xytext=(x_com[impact_index], y_com[impact_index]),  # arrow head
    xy=(x_test, y_test),                        # arrow start
    arrowprops=dict(
        arrowstyle='->',
        color='r',
        linewidth=1
    )
)
plt.annotate(
    f'({x_dot_com[impact_index]:.2f}, {y_dot_com[impact_index]:.2f})',
    xy=(x_test, y_test),
    xytext=(-20, 5),
    textcoords='offset points',
    fontsize=10
)


plt.annotate(
    '',
    xytext=(pend_x[impact_index], pend_y[impact_index]),  # arrow head
    xy=(pend_x_text, pend_y_text),                        # arrow start
    arrowprops=dict(
        arrowstyle='->',
        color='b',
        linewidth=1
    )
)
plt.annotate(
    f'({pend_x_dot[impact_index]:.2f}, {pend_y_dot[impact_index]:.2f})',
    xy=(pend_x_text, pend_y_text),
    xytext=(-20, 5),
    textcoords='offset points',
    fontsize=10
)

plt.annotate(
    '',
    xytext=(x_sim[impact_index], y_sim[impact_index]),  # arrow start
    xy=(hoop_x_test, hoop_y_test),                        # arrow head
    arrowprops=dict(
        arrowstyle='->',
        color='g',
        linewidth=1
    )
)
plt.annotate(
    f'({x_dot_sim[impact_index]:.2f}, {y_dot_sim[impact_index]:.2f})',
    xy=(hoop_x_test, hoop_y_test),
    xytext=(-20, 5),
    textcoords='offset points',
    fontsize=10
)
circle = Circle(
    (x_sim[impact_index], y_sim[impact_index]),
    0.1,
    fill=False,
    edgecolor='r'
)
plt.gca().add_patch(circle)

plt.xlabel("x")
plt.ylabel("y")
plt.title("AfterImpact: Impact and Test Point")
plt.axis("equal")
plt.grid(True)
plt.legend()
# plt.show()



#########################
#BEFORE IMPACT

########################
before_index= impact_index-1
plt.figure(figsize=(12,15))
plt.plot([x_sim[before_index], pend_x[before_index]], [y_sim[before_index], pend_y[before_index]], 'k-', linewidth=1, label='Impact')
plt.plot(x_com[before_index], y_com[before_index], 'o', markersize=8, label='Impact COM')
plt.plot(pend_x[before_index], pend_y[before_index], 'o', markersize=8, label='Impact Pendulum')
plt.plot(x_sim[before_index], y_sim[before_index], 'o', markersize=8, label='Impact Hoop Center')


# plt.plot([x_test, x_com[impact_index]], [y_test, y_com[impact_index]], 'r-', linewidth=1, label='Test Point')
plt.annotate(
    '',
    xytext=(x_com[before_index], y_com[before_index]),  # arrow head
    xy=(x_test_before, y_test_before),                        # arrow start
    arrowprops=dict(
        arrowstyle='->',
        color='r',
        linewidth=1
    )
)
plt.annotate(
    f'({x_dot_com[before_index]:.2f}, {y_dot_com[before_index]:.2f})',
    xy=(x_test_before, y_test_before),
    xytext=(-20, 5),
    textcoords='offset points',
    fontsize=10
)


plt.annotate(
    '',
    xytext=(pend_x[before_index], pend_y[before_index]),  # arrow head
    xy=(pend_x_text_before, pend_y_text_before),                        # arrow start
    arrowprops=dict(
        arrowstyle='->',
        color='b',
        linewidth=1
    )
)
plt.annotate(
    f'({pend_x_dot[before_index]:.2f}, {pend_y_dot[before_index]:.2f})',
    xy=(pend_x_text_before, pend_y_text_before),
    xytext=(-20, 5),
    textcoords='offset points',
    fontsize=10
)

plt.annotate(
    '',
    xytext=(x_sim[before_index], y_sim[before_index]),  # arrow start
    xy=(hoop_x_test_before, hoop_y_test_before),                        # arrow head
    arrowprops=dict(
        arrowstyle='->',
        color='g',
        linewidth=1
    )
)
plt.annotate(
    f'({x_dot_sim[before_index]:.2f}, {y_dot_sim[before_index]:.2f})',
    xy=(hoop_x_test_before, hoop_y_test_before),
    xytext=(-20, 5),
    textcoords='offset points',
    fontsize=10
)
circle = Circle(
    (x_sim[before_index], y_sim[before_index]),
    0.1,
    fill=False,
    edgecolor='r'
)
plt.gca().add_patch(circle)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Before Impact: Impact and Test Point")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(7, 6))
plt.plot(time, omega_rel_sim, linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("omega_rel [rad/s]")
plt.title("Forward Simulation: omega_rel vs Time")
plt.grid(True)
plt.show()
plt.figure(figsize=(7, 6))
plt.plot(time, omega_rel_sim, linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("omega_rel [rad/s]")
plt.title("Forward Simulation: omega_rel vs Time")
plt.grid(True)
plt.show()



################

plt.figure(figsize=(7, 6))
plt.subplot(4, 1, 1)
plt.plot(time, phidot_sim, linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("phi_dot [rad/s]")
plt.title("Forward Simulation: phi_dot vs Time")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(time, thetadot_sim, linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("theta_dot [rad/s]")
plt.title("Forward Simulation: theta_dot vs Time")
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(time, phi_sim, linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("phi [rad]")
plt.title("Forward Simulation: phi vs Time")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(time, np.rad2deg(theta_sim), linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("theta [deg]")
plt.title("Forward Simulation: theta vs Time")
plt.grid(True)

plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 6))
plt.subplot(4, 1, 1)
plt.plot(time, x_dot_com, linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("x_dot_com [m/s]")
plt.title("Forward Simulation: x_dot_com vs Time")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(time, y_dot_com, linewidth=2)
plt.axvline(impact_time, color='r', linestyle='--', label='Impact Time')

plt.xlabel("Time [s]")
plt.ylabel("y_dot_com [m/s]")
plt.title("Forward Simulation: y_dot_com vs Time")
plt.grid(True)


plt.subplot(4, 1, 3)
plt.plot(time, x_dot_sim, linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("x_dot_sim [m/s]")
plt.title("Forward Simulation: x_dot_sim vs Time")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(time, y_dot_sim, linewidth=2)
plt.axvline(impact_time, color='r', linestyle='--', label='Impact Time')
plt.xlabel("Time [s]")
plt.ylabel("y_dot_sim [m/s]")
plt.title("Forward Simulation: y_dot_sim vs Time")
plt.grid(True)


plt.tight_layout()
plt.show()

