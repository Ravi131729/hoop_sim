import numpy as np
import casadi as ca

from quantities import center_of_mass_position, center_of_mass_velocity, pendulum_position, pendulum_velocity,hoop_position_relative_to_com
from parameters import ModelParams
from dynamics import build_dynamics, impact_map
from plotting import plot_reset_map,plot_trajectory,plot_angular_momentum
f, M_fun = build_dynamics()
p_val = ModelParams().as_array()
MAX_SPEED = 200
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

# N = 1000
T = 0.5
dt = 0.001

N = int(T/dt)
nx = 8
nu = 1
y_contact = 0.12
x0 = np.array([0.0, 0.3, 0.0, 0.0,
               1.0, 0.0, 10.0, 0.0])


u_val = np.array([0.9])

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

        )
        print("qdot_plus =", qdot_plus)

        x_new[1] = y_contact
        x_new[4:8] = qdot_plus

        x_plus = x_new.copy()

        print("Impact mode:", mode)
        print("lambda_t =", lambda_t)
        print("lambda_N =", lambda_n)
        count += 1
        if count ==1:
          pre_impact_state = x_minus
          post_impact_state = x_plus
        print(f"Number of impacts: {count}")
    Xsim[k + 1, :] = x_new


print("Simulation complete.")

hoop_pos_rcom  = [hoop_position_relative_to_com(Xsim[i, :], p_val) for i in range(Xsim.shape[0])]

print("hoop_pos_rcom =", np.linalg.norm(hoop_pos_rcom,axis=1))

plot_reset_map(pre_impact_state, post_impact_state, p_val)

plot_trajectory(Xsim, p_val, impact_index)
plot_angular_momentum(Xsim, p_val, impact_index)