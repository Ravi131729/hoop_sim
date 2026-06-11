# General Dynamics and Controller Formulation

Consider a multibody robotic system with generalized configuration \(q\), generalized velocity \(v\), and generalized acceleration \(\dot v\). Its continuous-time rigid-body dynamics can be written as

\[
M(q)\dot v + h(q,v) = S^T\tau + J_c(q)^T\lambda ,
\]

where \(M(q)\) is the mass matrix, \(h(q,v)\) contains Coriolis, centrifugal, and gravity terms, \(\tau\) is the vector of actuator torques, \(S\) maps actuator torques into generalized coordinates, \(J_c\) is the contact Jacobian, and \(\lambda\) is the contact wrench or contact force. If the system has no active contact, the contact term is removed:

\[
M(q)\dot v + h(q,v) = S^T\tau .
\]

For a task frame or output \(x=f(q)\), the task velocity and acceleration are

\[
\dot x = J(q)v,
\qquad
\ddot x = J(q)\dot v + \dot J(q,v)v .
\]

The controller specifies a desired task acceleration \(\ddot x_{\mathrm{des}}\). A common choice is a proportional-derivative law in task space,

\[
\ddot x_{\mathrm{des}}
=
\ddot x_d
-K_p e_x
-K_d e_{\dot x},
\]

where \(e_x\) is the task-space position or orientation error and \(e_{\dot x}\) is the task velocity error. For orientation control on \(SO(3)\), the attitude error is usually computed with the Lie algebra logarithm,

\[
e_R = \log(RR_d^T),
\qquad
e_\omega = \omega-\omega_d,
\]

and the desired angular acceleration is

\[
\alpha_{\mathrm{des}}
=
-K_R e_R - K_\omega e_\omega .
\]

The whole-body controller is formulated as an inverse-dynamics quadratic program. The decision variables are typically generalized acceleration, actuator torque, and optionally contact force:

\[
z =
\begin{bmatrix}
\dot v \\
\tau \\
\lambda
\end{bmatrix}.
\]

The QP minimizes task acceleration error while satisfying the robot dynamics:

\[
\begin{aligned}
\min_{\dot v,\tau,\lambda}\quad
&
\frac{1}{2}
\left(J\dot v+\dot Jv-\ddot x_{\mathrm{des}}\right)^T
W
\left(J\dot v+\dot Jv-\ddot x_{\mathrm{des}}\right)
\\
&
+\frac{1}{2}\dot v^T W_{\dot v}\dot v
+\frac{1}{2}\tau^T W_\tau\tau
+\frac{1}{2}\lambda^T W_\lambda\lambda
\\[2mm]
\text{subject to}\quad
&
M(q)\dot v+h(q,v)=S^T\tau+J_c(q)^T\lambda ,
\\
&
\tau_{\min}\le \tau\le \tau_{\max},
\\
&
\lambda \in \mathcal{K}.
\end{aligned}
\]

Here \(W\) is the task weight matrix, and \(W_{\dot v}\), \(W_\tau\), and \(W_\lambda\) are regularization weights. The set \(\mathcal{K}\) represents contact-force constraints, such as unilateral normal force and friction-cone limits. For example, a simple point contact may require

\[
\lambda_n \ge 0,
\qquad
\sqrt{\lambda_t^T\lambda_t} \le \mu \lambda_n ,
\]

where \(\lambda_n\) is the normal force, \(\lambda_t\) is the tangential friction force, and \(\mu\) is the friction coefficient.

If a contact point is assumed to remain fixed, its acceleration must satisfy

\[
J_c\dot v+\dot J_c v = 0 .
\]

With stabilization, this can be written as

\[
J_c\dot v+\dot J_c v
=
-K_p^c e_c - K_d^c \dot e_c ,
\]

where \(e_c\) and \(\dot e_c\) are contact position and velocity errors. This prevents numerical drift in the contact constraint.

After solving the QP, the controller applies the optimized torque \(\tau^\star\). In simulation, the optimized acceleration \(\dot v^\star\) can also be integrated forward:

\[
v_{k+1}=v_k+\Delta t\,\dot v^\star,
\qquad
q_{k+1}=\operatorname{integrate}(q_k,\Delta t\,v_{k+1}).
\]

This type of controller is called a whole-body quadratic-programming controller, or inverse-dynamics QP controller. Its main advantage is that it combines task tracking, rigid-body dynamics, torque limits, and contact constraints in a single optimization problem. By changing the task Jacobian \(J\), task weights \(W\), and constraints, the same formulation can be used for attitude control, end-effector tracking, center-of-mass control, balancing, or contact-rich locomotion.

Contact constraints are needed whenever the robot interacts with the environment through feet, wheels, hands, or any other support point. Without these constraints, the optimizer may choose accelerations and torques that satisfy the desired motion mathematically but are physically impossible, such as moving through the ground, pulling on a surface, or generating tangential forces beyond friction limits. Contact constraints make the solution respect non-penetration, support forces, and friction, so the resulting motion is dynamically feasible and consistent with the environment.
