import numpy as np


def center_of_mass_position(x, p):
    mp = p[1]
    mo = p[2]
    Lp = p[3]
    theta = x[3]

    total_mass = mo + mp

    x_com = (
        total_mass * x[0]
        - mp * Lp * np.sin(theta)
    ) / total_mass

    y_com = (
        total_mass * x[1]
        - mp * Lp * np.cos(theta)
    ) / total_mass

    return np.array([x_com, y_com])


def center_of_mass_velocity(x, p):
    dx = x[4]
    dy = x[5]
    dtheta = x[7]

    mp = p[1]
    mo = p[2]
    Lp = p[3]
    theta = x[3]

    total_mass = mo + mp

    dx_com = (
        total_mass * dx
        - mp * Lp * np.cos(theta) * dtheta
    ) / total_mass

    dy_com = (
        total_mass * dy
        + mp * Lp * np.sin(theta) * dtheta
    ) / total_mass

    return np.array([dx_com, dy_com])


def pendulum_position(x, p):
    Lp = p[3]
    theta = x[3]

    pend_x = x[0] - Lp * np.sin(theta)
    pend_y = x[1] - Lp * np.cos(theta)

    return np.array([pend_x, pend_y])


def pendulum_velocity(x, p):
    dx = x[4]
    dy = x[5]
    dtheta = x[7]

    Lp = p[3]
    theta = x[3]

    pend_vel_x = dx - Lp * np.cos(theta) * dtheta
    pend_vel_y = dy + Lp * np.sin(theta) * dtheta

    return np.array([pend_vel_x, pend_vel_y])


def hoop_velocity_relative_to_com(x, p):
    v_com = center_of_mass_velocity(x, p)

    v_hoop = np.array([
        x[4],
        x[5]
    ])

    return v_hoop - v_com


def pendulum_velocity_relative_to_com(x, p):
    v_pend = pendulum_velocity(x, p)
    v_com = center_of_mass_velocity(x, p)

    return v_pend - v_com


def hoop_position_relative_to_com(x, p):
    r_com = center_of_mass_position(x, p)

    r_hoop = np.array([
        x[0],
        x[1]
    ])

    return r_hoop - r_com


def pendulum_position_relative_to_com(x, p):
    r_pend = pendulum_position(x, p)
    r_com = center_of_mass_position(x, p)

    return r_pend - r_com


def hoop_angular_momentum_com(x, p):
    Io = p[0]
    mo = p[2]

    dphi = x[6]

    r = hoop_position_relative_to_com(x, p)
    v = hoop_velocity_relative_to_com(x, p)

    H_hoop_com = (
        Io * dphi

        - mo * (
            r[0] * v[1]
            - r[1] * v[0]
        )
    )

    return H_hoop_com


def pendulum_angular_momentum_com(x, p):
    mp = p[1]

    r = pendulum_position_relative_to_com(x, p)
    v = pendulum_velocity_relative_to_com(x, p)

    H_pend_com = -mp * (
        r[0] * v[1]
        - r[1] * v[0]
    )

    return H_pend_com


def angular_momentum(x, p):
    H_hoop = hoop_angular_momentum_com(x, p)
    H_pend = pendulum_angular_momentum_com(x, p)

    return H_hoop + H_pend