# Dynamics and Controller Write-Up

This repository simulates a hoop/rotor rigid-body system using Pinocchio for multibody dynamics. The generalized configuration and velocity are denoted by \(q\) and \(v\), with dimension `model.nv`. At each simulation step, `ran_test.py` computes forward kinematics, frame placements, joint Jacobians, the mass matrix \(M(q)\), and the nonlinear bias term \(h(q,v)\), where \(h\) contains Coriolis, centrifugal, and gravity effects. The unconstrained rigid-body dynamics are

\[
M(q)\dot v + h(q,v) = \tau ,
\]

where \(\tau\) is the vector of generalized actuation forces. In the passive/contact simulation, most torques are zero except for a small commanded internal torque on the pendulum/rotor coordinate.

Ground contact is modeled at the lowest point of the hoop. If \(c\) is the hoop center, \(r\) is the hoop radius, \(R_{wb}\) is the hoop orientation, and \(n_g=[0,0,1]^T\) is the ground normal, the hoop plane normal is

\[
n_h = R_{wb}[0,1,0]^T .
\]

The ground normal is projected into the hoop plane,

\[
n_\parallel = n_g - (n_g^T n_h)n_h ,
\]

and the contact direction is \(d=-n_\parallel/\|n_\parallel\|\) when this projection is nonzero. The contact point is then

\[
p_c = c + r d, \qquad r_c = p_c - c .
\]

The code forms a contact Jacobian from the hoop frame linear and angular Jacobians \(J_v\) and \(J_\omega\):

\[
J_c = J_v - [r_c]_\times J_\omega ,
\]

so that the contact point velocity is \(v_c = J_c v\). Contact is enforced as an acceleration-level constraint with Baumgarte stabilization,

\[
J_c\dot v = -\dot J_c v - \alpha v_c ,
\]

where \(\alpha=10\) in the simulation. With contact force \(\lambda\), the constrained equations are

\[
M\dot v + h = \tau + J_c^T\lambda .
\]

The free acceleration is \(\dot v_{\text{free}}=M^{-1}(\tau-h)\). Substituting the dynamics into the contact constraint gives the linear system

\[
\left(J_cM^{-1}J_c^T\right)\lambda
=
-\dot J_c v - \alpha v_c - J_c\dot v_{\text{free}} .
\]

After solving for \(\lambda\), the constrained acceleration is

\[
\dot v = \dot v_{\text{free}} + M^{-1}J_c^T\lambda .
\]

The simulation advances the state using semi-implicit Euler integration:

\[
v_{k+1}=v_k+\Delta t\,\dot v_k,\qquad
q_{k+1}=\operatorname{integrate}(q_k,\Delta t\,v_{k+1}),
\]

with \(\Delta t=10^{-4}\) s. Pinocchio's `integrate` is used so that rotations and floating-base coordinates remain on the correct configuration manifold.

The attitude controller in `attitude_controller.py` is implemented as a whole-body quadratic program. The desired attitude is \(R_d\), the current hoop-frame attitude is \(R\), and the angular velocity is \(\omega\). The rotation error is computed in the world frame as

\[
e_R = \log\left(RR_d^T\right),
\qquad
e_\omega = \omega-\omega_d .
\]

A PD law defines the desired angular acceleration,

\[
\alpha_{\text{des}} = -k_R e_R - k_\omega e_\omega ,
\]

with \(k_R=400\) and \(k_\omega=32\). The controller tracks only the angular part of the six-dimensional frame acceleration. The decision variable is

\[
x = \begin{bmatrix}\dot v \\\tau\end{bmatrix}.
\]

The QP minimizes angular acceleration tracking error plus a small acceleration regularization:

\[
\min_{\dot v,\tau}
\frac{1}{2}
\left(J\dot v+\dot Jv-\ddot x_{\text{des}}\right)^T
W
\left(J\dot v+\dot Jv-\ddot x_{\text{des}}\right)
+ \frac{\lambda}{2}\|\dot v\|^2
+ \frac{w_\tau}{2}\|\tau\|^2 ,
\]

where \(W\) weights only angular rows, \(\ddot x_{\text{des}}=[0,0,0,\alpha_{\text{des}}]^T\), and \(\lambda=3\times10^{-4}\). The equality constraint enforces rigid-body dynamics,

\[
M(q)\dot v-\tau=-h(q,v),
\]

and box constraints limit actuation. In the current code, all generalized torques are fixed to zero except the last two actuated coordinates, with bounds \([-5,5]\) and \([-10,10]\). OSQP solves this convex QP and returns the commanded generalized acceleration and torque.
