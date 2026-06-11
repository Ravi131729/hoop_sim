# Forward Dynamics Simulation with Rolling Constraints

This note describes how to simulate a rigid-body system forward in time when the unconstrained dynamics are known, and then how to add rolling/contact constraints using a KKT system.

## 1. Unconstrained Forward Dynamics

For this system, let the generalized configuration contain the position and orientation of the hoop:

\[
q =
\begin{bmatrix}
p \\
R\\
\theta_p\\
\theta_r
\end{bmatrix},
\]

where \(p\in\mathbb{R}^3\) is the position of the hoop frame and \(R\in SO(3)\) is the rotation matrix that represents the hoop orientation. The generalized velocity is

\[
\dot q =
\begin{bmatrix}
v \\
\omega\\
\dot \theta_p\\
\dot \theta_r
\end{bmatrix},
\]

where \(v\in\mathbb{R}^3\) is the linear velocity of the hoop frame and \(\omega\in\mathbb{R}^3\) is the angular velocity.

For a multibody system with generalized configuration \(q\), generalized velocity \(\dot q\), and generalized acceleration \(\ddot q\), the rigid-body dynamics are

\[
M(q)\ddot q + h(q,\dot q) = \tau ,
\]

where \(M(q)\) is the mass matrix, \(h(q,\dot q)\) contains Coriolis, centrifugal, and gravity effects, and \(\tau\) is the generalized force/torque vector.

If there is no contact or constraint, the forward dynamics are obtained directly:

\[
\ddot q = M(q)^{-1}\left(\tau-h(q,\dot q)\right).
\]

Then the state is integrated forward:

\[
\dot q_{k+1}=\dot q_k+\Delta t\,\ddot q_k,
\]

\[
q_{k+1}=\operatorname{integrate}(q_k,\Delta t\,\dot q_{k+1}),
\]

which means

\[
p_{k+1}=p_k+\Delta t\,v_{k+1},
\]

\[
R_{k+1}=R_k\exp\left([\omega_{k+1}]_\times\Delta t\right).
\]

Here \([\omega]_\times\) is the skew-symmetric matrix form of angular velocity:

\[
[\omega]_\times =
\begin{bmatrix}
0 & -\omega_z & \omega_y \\
\omega_z & 0 & -\omega_x \\
-\omega_y & \omega_x & 0
\end{bmatrix}.
\]

This rotation-matrix update keeps \(R_{k+1}\) on \(SO(3)\), instead of adding directly to the entries of \(R\).

## 2. Rolling Constraint

For rolling contact, the contact point on the body should have zero velocity relative to the ground. Let \(p_c(q)\) be the contact point and let its velocity be

\[
\dot r_c = J_c(q)\dot q ,
\]

where \(J_c(q)\) is the contact Jacobian.

For pure rolling without slipping on a stationary surface, the velocity constraint is

\[
J_c(q)\dot q = 0 .
\]

To use the constraint in forward dynamics, differentiate it:

\[
\frac{d}{dt}\left(J_c \dot q\right)=0 .
\]

Therefore,

\[
J_c\ddot q + \dot J_c \dot q = 0 .
\]

This is the acceleration-level rolling/contact constraint.


## 3. Dynamics with Constraint Forces

The constrained forward dynamics can be written as a saddle-point or KKT system:

\[
\begin{bmatrix}
M(q) & -J_c(q)^T \\
J_c(q) & 0
\end{bmatrix}
\begin{bmatrix}
\ddot q \\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
\tau-h(q,\dot q) \\
-\dot J_c(q,\dot q)\dot q
\end{bmatrix}.
\]

## 4. How to Build the Contact Jacobian

Suppose the body frame has linear Jacobian \(J_v\) and angular Jacobian \(J_\omega\). Let \(r_c\) be the vector from the body origin or center of mass to the contact point.

A rigid body twist contains two different kinds of motion:

\[
\begin{bmatrix}
v_{\text{hoop}} \\
\omega
\end{bmatrix}
=
J
\dot q .
\]
\[
\begin{bmatrix}
v_{\text{hoop}} \\
\omega
\end{bmatrix}
=
\begin{bmatrix}
J_v \\
J_\omega
\end{bmatrix}
\dot q .
\]
The linear part \(J_v\dot q\) gives the translational velocity of the body frame origin. The angular part \(J_\omega\dot q\) gives the angular velocity of the body. We split the twist this way because a point away from the body origin does not only move with the origin; it also moves because the body rotates around the origin. That rotational contribution is \(\omega \times r_c\), so the contact point velocity needs both pieces.

\[
v_c = v_{\text{hoop}} + \omega \times r_c .
\]

In Jacobian form,

\[
v_c = \left(J_v - [r_c]_\times J_\omega\right)\dot q .
\]

Therefore, the contact Jacobian is

\[
J_c = J_v - [r_c]_\times J_\omega .
\]

This maps generalized velocity to the Cartesian velocity of the contact point.

## 5. Derivative of the Contact Velocity
The contact-point  acceleration can then be formed as

\[
\dot J_c \dot q
=
a_{\text{hoop}}
+ \dot \omega \times r_c
+ \omega \times (\dot r_c),
\]

\[
\dot J_c \dot q
=
a_{\text{hoop}}
+ \dot \omega \times r_c
+ \textcolor{red}{\omega \times (\omega \times r_c)}
\]

Once the contact Jacobian is known, the contact point velocity is

\[
v_c = J_c(q)\dot q .
\]

To impose a rolling or no-slip constraint in acceleration form, take the time derivative:

\[
\dot v_c
=
\frac{d}{dt}\left(J_c(q)\dot q\right).
\]

Using the product rule,

\[
\dot v_c
=
J_c(q)\ddot q + \dot J_c(q,\dot q)\dot q .
\]

This is why the acceleration-level contact constraint contains two terms:

\[
J_c\ddot q+\dot J_c \dot q=0 .
\]



## 6. Forward Simulation Algorithm

At each simulation step:

1. Compute kinematics from the current state:

\[
q_k,\ \dot q_k .
\]

2. Compute rigid-body dynamics terms:

\[
M(q_k),\qquad h(q_k,\dot q_k).
\]

3. Compute the contact point and contact Jacobian:

\[
r_c(q_k),\qquad J_c(q_k).
\]

4. Compute the contact velocity:

\[
v_c = J_c \dot q_k .
\]

5. Compute or approximate the bias acceleration term:

\[
\dot J_c \dot q_k .
\]

6. Build the KKT system:

\[
\begin{bmatrix}
M & -J_c^T \\
J_c & 0
\end{bmatrix}
\begin{bmatrix}
\ddot q \\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
\tau-h \\
-\dot J_c \dot q - \alpha v_c
\end{bmatrix}.
\]

7. Solve for \(\ddot q\) and \(\lambda\).

8. Integrate:

\[
\dot q_{k+1}=\dot q_k+\Delta t\,\ddot q,
\]

\[
q_{k+1}=\operatorname{integrate}(q_k,\Delta t\,\dot q_{k+1}).
\]

9. Repeat until the final simulation time.

## 8. Contact Validity

The KKT system assumes the contact remains active. In a complete simulator, this assumption should be checked. A unilateral contact can only push, not pull, so the normal force should satisfy

\[
\lambda_n \ge 0 .
\]

If \(\lambda_n < 0\), the contact should be removed and the system should switch to unconstrained flight dynamics. For rolling with friction, tangential forces should also satisfy the friction cone:

\[
\|\lambda_t\| \le \mu \lambda_n .
\]

If this condition fails, the pure rolling assumption is no longer valid and the contact should switch to a sliding model or a friction-limited model.
