import numpy as np


NX = 8
NU = 1
NP = 5


def mass_matrix(X, P):
    theta = X[3]

    Io = P[0]
    mp = P[1]
    mo = P[2]
    Lp = P[3]

    M = np.array([
        [
            mo + mp,
            0.0,
            0.0,
            -Lp * mp * np.cos(theta),
        ],
        [
            0.0,
            mo + mp,
            0.0,
            Lp * mp * np.sin(theta),
        ],
        [
            0.0,
            0.0,
            Io,
            0.0,
        ],
        [
            -Lp * mp * np.cos(theta),
            Lp * mp * np.sin(theta),
            0.0,
            Lp**2 * mp,
        ],
    ])

    return M


def force_vector(X, U, P):
    theta = X[3]
    dtheta = X[7]

    mp = P[1]
    mo = P[2]
    Lp = P[3]
    g = P[4]

    tau = U[0]

    F = np.array([
        -mp * Lp * np.sin(theta) * dtheta**2,

        -mp * Lp * dtheta**2 * np.cos(theta)
        - (mo + mp) * g,

        -tau,

        tau - mp * Lp * g * np.sin(theta),
    ])

    return F


def dynamics(X, U, P):
    dx = X[4]
    dy = X[5]
    dphi = X[6]
    dtheta = X[7]

    M = mass_matrix(X, P)
    F = force_vector(X, U, P)

    # Equivalent to CasADi:
    # qdd = ca.solve(M, F)
    qdd = np.linalg.solve(M, F)

    xdot = np.concatenate([
        np.array([
            dx,
            dy,
            dphi,
            dtheta,
        ]),
        qdd,
    ])

    return xdot


def build_dynamics():
    return dynamics, mass_matrix

def impact_map(qdot_minus, q, p):
    e = p[6]
    mu = p[7]
    R= p[5]
    # Mass matrix at impact
    M_num = mass_matrix(q, p)

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

# f, M_fun = build_dynamics()