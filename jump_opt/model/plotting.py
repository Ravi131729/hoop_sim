import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from quantities import center_of_mass_position, pendulum_position, pendulum_velocity, center_of_mass_velocity, angular_momentum


def plot_reset_map(preimpact_state, postimpact_state, parameters, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Plot pre-impact state
    x_pre = preimpact_state[0]
    y_pre = preimpact_state[1]
    theta_pre = preimpact_state[3]

    # Plot post-impact state
    x_post = postimpact_state[0]
    y_post = postimpact_state[1]
    theta_post = postimpact_state[3]

    # Plot the hoop and pendulum for pre-impact state
    hoop_radius = parameters[5]  # Assuming R is the 6th parameter
    pendulum_length = parameters[3]  # Assuming Lp is the 4th parameter



    #compute all the positions and velocities
    com_pre = center_of_mass_position(preimpact_state, parameters)
    com_post = center_of_mass_position(postimpact_state, parameters)
    com_vel_pre = center_of_mass_velocity(preimpact_state, parameters)
    com_vel_post = center_of_mass_velocity(postimpact_state, parameters)

    #pendulum positions and velocities
    pend_vel_pre = pendulum_velocity(preimpact_state, parameters)
    pend_vel_post = pendulum_velocity(postimpact_state, parameters)
    pendulum_pos_pre = pendulum_position(preimpact_state, parameters)
    pendulum_pos_post = pendulum_position(postimpact_state, parameters)

    #hoop positions and velocities
    hoop_vel_pre = np.array([preimpact_state[4], preimpact_state[5]])
    hoop_vel_post = np.array([postimpact_state[4], postimpact_state[5]])
    hoop_pos_pre = np.array([preimpact_state[0], preimpact_state[1]])
    hoop_pos_post = np.array([postimpact_state[0], postimpact_state[1]])




    plt.subplot(1, 2, 1)
    plt.gca().add_patch(Circle((x_pre, y_pre), hoop_radius, fill=False, label='Hoop (Pre-impact)'))
    plt.plot([x_pre, pendulum_pos_pre[0]], [y_pre, pendulum_pos_pre[1]], label='Pendulum (Pre-impact)')
    plt.plot(pendulum_pos_pre[0], pendulum_pos_pre[1], 'ro')  # Pendulum tip
    plt.plot(com_pre[0], com_pre[1], 'go', label='COM (Pre-impact)')  # Center of mass
    plt.plot(hoop_pos_pre[0], hoop_pos_pre[1], 'bo', label='Hoop (Pre-impact)')  # Hoop position
    plt.quiver(com_pre[0], com_pre[1], com_vel_pre[0], com_vel_pre[1],  scale=15, label='COM Velocity (Pre-impact)', width=0.003,color='green')
    plt.quiver(pendulum_pos_pre[0], pendulum_pos_pre[1], pend_vel_pre[0], pend_vel_pre[1],  scale=15, label='Pendulum Velocity (Pre-impact)', width=0.003,color='red')
    plt.quiver(hoop_pos_pre[0], hoop_pos_pre[1], hoop_vel_pre[0], hoop_vel_pre[1],  scale=15, label='Hoop Velocity (Pre-impact)', width=0.003,color='blue')



    plt.title('Pre-impact State')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid()

    # Plot the hoop and pendulum for post-impact state
    plt.subplot(1, 2, 2)
    plt.gca().add_patch(Circle((x_post, y_post), hoop_radius, fill=False,  label='Hoop (Post-impact)'))
    plt.plot([x_post, pendulum_pos_post[0]], [y_post, pendulum_pos_post[1]],  label='Pendulum (Post-impact)')
    plt.plot(pendulum_pos_post[0], pendulum_pos_post[1], 'ro')  # Pendulum tip
    plt.plot(com_post[0], com_post[1], 'go', label='COM (Post-impact)')  # Center of mass
    plt.plot(hoop_pos_post[0], hoop_pos_post[1], 'bo', label='Hoop (Post-impact)')  # Hoop position
    plt.quiver(com_post[0], com_post[1], com_vel_post[0], com_vel_post[1],  scale=15, label='COM Velocity (Post-impact)', width=0.003,color='green')
    plt.quiver(pendulum_pos_post[0], pendulum_pos_post[1], pend_vel_post[0], pend_vel_post[1],  scale=15, label='Pendulum Velocity (Post-impact)', width=0.003,color='red')
    plt.quiver(hoop_pos_post[0], hoop_pos_post[1], hoop_vel_post[0], hoop_vel_post[1],  scale=15, label='Hoop Velocity (Post-impact)', width=0.003,color='blue')

    plt.title('Post-impact State')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_trajectory(X, parameters, impact_index,ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Extract positions
    x_positions = X[:, 0]
    y_positions = X[:, 1]

    y_contact = parameters[5]  # Assuming y_contact is the 5th parameter
    x_impact = X[impact_index, 0]
    y_impact = X[impact_index, 1]

    com_x_impact = center_of_mass_position(X[impact_index, :], parameters)[0]
    com_y_impact = center_of_mass_position(X[impact_index, :], parameters)[1]

    pend_x_impact = pendulum_position(X[impact_index, :], parameters)[0]
    pend_y_impact = pendulum_position(X[impact_index, :], parameters)[1]

    com_pos = np.array([center_of_mass_position(X[i, :], parameters) for i in range(X.shape[0])])
    com_x_positions = com_pos[:, 0]
    com_y_positions = com_pos[:, 1]

    # Plot trajectory
    plt.plot(x_positions, y_positions, label='Trajectory', linewidth=2)
    plt.plot(com_x_positions, com_y_positions, label='Center of Mass Trajectory', linewidth=2, linestyle='--')
    plt.plot([x_impact, pend_x_impact], [y_impact, pend_y_impact], 'r', label='Pendulum at Impact')
    plt.plot(x_impact, y_impact, 'bo', label='Impact Point')
    plt.plot(com_x_impact, com_y_impact, 'go', label='COM at Impact')
    plt.plot(pend_x_impact, pend_y_impact, 'ro', label='Pendulum Tip at Impact')
    plt.axhline(y=y_contact, color='k', linestyle='--', label='Contact Line (y = {:.2f})'.format(y_contact))
    plt.gca().add_patch(Circle((x_impact, y_impact), parameters[5], color='orange', fill=False, label='Impact Circle'))
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Hoop-Pendulum Trajectory')
    plt.axis('equal')
    plt.grid()
    # plt.legend()
    plt.show()

def plot_angular_momentum(X, parameters, impact_index, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Compute angular momentum at each time step
    H_com = np.array([angular_momentum(X[i, :], parameters) for i in range(X.shape[0])])

    # Plot angular momentum
    plt.plot(H_com, label='Angular Momentum about COM', linewidth=2)
    plt.axvline(x=impact_index, color='r', linestyle='--', label='Impact Event')
    plt.xlabel('Time Step')
    plt.ylabel('Angular Momentum (kg*m^2/s)')
    plt.title('Angular Momentum about Center of Mass')
    plt.grid()
    plt.legend()
    plt.show()