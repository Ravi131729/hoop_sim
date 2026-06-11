import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import time
from collision_test import spinjorr
import meshcat
import meshcat_shapes
from utils import hoop_contact_point
import matplotlib.pyplot as plt
import meshcat.geometry as mg
import scipy



def get_rtop(R,normal):

    n_h = R.T @ normal
    # print(n_h)

    alpha = np.arctan2(n_h[0], n_h[2])
    rho = 0.1

    r_top_h = np.array([ rho * np.sin(alpha), 0.0, rho * np.cos(alpha) ])
    r_top = R @ r_top_h

    return r_top,alpha


def get_alpha_dot(R, omega, normal):

    omega_h = R.T @ omega
    n_h = R.T @ normal

    x, y, z = n_h

    # n_dot = n × omega
    n_dot = np.cross(n_h, omega_h)

    x_dot = n_dot[0]
    z_dot = n_dot[2]

    alpha_dot = (z * x_dot - x * z_dot) / (x**2 + z**2)

    return alpha_dot


def get_rc_dot(R, omega, normal):

    omega_h = R.T @ omega
    r_top,alpha = get_rtop(R, normal)

    alpha_dot = get_alpha_dot(R, omega, normal)
    rho = 0.1
    rho_s = 0.125

    r_top_h = np.array([ rho * np.sin(alpha), 0.0, rho * np.cos(alpha) ])



    r_th_dot = R@np.cross(omega_h, r_top_h) + R @ np.array([ rho * np.cos(alpha) * alpha_dot, 0.0, -rho * np.sin(alpha) * alpha_dot ])
    # print("aplha_dot", alpha_dot)
    b = (rho_s - rho)/rho

    rc_dot = b*r_th_dot
    rc  = b*r_top - rho_s * normal

    return rc_dot,rc




hoop = spinjorr(visualize=True)
for i, f in enumerate(hoop.model.frames):
    print(i, f.name, f.parentJoint)
fid_hoop = hoop.model.addFrame(
    pin.Frame(
        "hoop_frame",                 # frame name
        1,          # parent joint
        0,                      # parent frame index (0 = universe, OK here)
        pin.SE3.Identity(),     # placement in parent joint frame
        pin.FrameType.OP_FRAME
    )
)
fid_rotor = hoop.model.addFrame(
    pin.Frame(
        "rotor_frame",                 # frame name
        3,
        2,          # parent joint
        pin.SE3.Identity(),     # placement in parent joint frame                     # parent frame index (0 = universe, OK here)
        pin.FrameType.OP_FRAME
    )
)
meshcat_shapes.frame(
    hoop.viz.viewer["rotor_frame"],
    axis_length=0.2,
    axis_thickness=0.005,
    opacity=0.8,
    origin_radius=0.002,
)
meshcat_shapes.frame(
    hoop.viz.viewer["hoop_frame"],
    axis_length=0.2,
    axis_thickness=0.005,
    opacity=0.8,
    origin_radius=0.002,
)



# --- Initial configuration ---
q = pin.neutral(hoop.model)
v= np.zeros(hoop.model.nv)
theta = np.deg2rad(00.0)
R0 = pin.utils.rotate('x', theta)
R1 = pin.utils.rotate('z', np.deg2rad(0.0))
quat = pin.Quaternion(R1@R0)   # Pinocchio quaternion

q[2] = 0.1
# q[-2]= -np.deg2rad(30.0)  # pendulum angle
# q[0] = 1
q[3:7] = quat.coeffs()

hoop.data = hoop.model.createData()

pin.forwardKinematics(hoop.model, hoop.data, q, v,np.zeros(hoop.model.nv))
pin.updateFramePlacements(hoop.model, hoop.data)
fid_rotor = hoop.model.getFrameId("rotor_frame")
fid_hoop  = hoop.model.getFrameId("hoop_frame")

oMfr = hoop.data.oMf[fid_rotor]   # SE3
oMfh = hoop.data.oMf[fid_hoop]

# Send to Meshcat
hoop.viz.viewer["rotor_frame"].set_transform(oMfr.homogeneous)
hoop.viz.viewer["hoop_frame"].set_transform(oMfh.homogeneous)

traj_node = hoop.viz.viewer["hoop_trajectory"]
trajectory = []
# print(q)
# --- Display ---
hoop.display(q)


dt = 1e-3
T  = 15
N  = int(T / dt)

# Constants
hoop_radius = 0.1
alpha_baumgarte = 20.0  # Baumgarte stabilization gain

# Data logging
times = []
contact_forces_x = []
contact_forces_y = []
contact_forces_z = []
contact_heights = []
contact_vel_x = []
contact_vel_y = []
contact_vel_z = []
hoop_pos_x = []
hoop_pos_y = []
hoop_pos_z = []
pend_velocity = []
hoop_velocity = []
pend_pos = []
rotor_pos = []
roll_list = []
pitch_list = []
yaw_list = []
theta_list = []
input_torque_list = []
t=0.0
for i in range(N):
  # Forward kinematics
  pin.forwardKinematics(hoop.model, hoop.data, q, v, np.zeros(hoop.model.nv))
  pin.updateFramePlacements(hoop.model, hoop.data)
  pin.computeJointJacobians(hoop.model, hoop.data, q)
  pin.crba(hoop.model, hoop.data, q)  # Compute mass matrix
  pin.nonLinearEffects(hoop.model, hoop.data, q, v)  # Compute bias (C*v + g)

  fid_rotor = hoop.model.getFrameId("rotor_frame")
  fid_hoop  = hoop.model.getFrameId("hoop_frame")

  oMfr = hoop.data.oMf[fid_rotor]
  oMfh = hoop.data.oMf[fid_hoop]

  # Frame Jacobian in LOCAL_WORLD_ALIGNED frame
  J = pin.computeFrameJacobian(hoop.model, hoop.data, q, fid_hoop, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
  Jdotv = pin.getFrameClassicalAcceleration(hoop.model, hoop.data, fid_hoop, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

  Jv = J[:3, :]    # linear velocity Jacobian
  Jw = J[3:, :]    # angular velocity Jacobian
  omega = Jw @ v

  # Get hoop center and rotation
  hoop_center = q[0:3].copy()
  R_wb = oMfh.rotation.copy()
  r_top,alpha = get_rtop(R_wb, np.array([0.0, 0.0, 1.0]))

  r_top = hoop_center + r_top

  rotor_pos_w = oMfr.translation.copy()
  hoop_pos_w = oMfh.translation.copy()
  theta = np.arctan2( hoop_center[0]-rotor_pos_w[0], hoop_center[2]-rotor_pos_w[2])
#   print("theta", np.rad2deg(theta))
#   print("hoop_center", hoop_center)
#   print("hoop_pos_w", hoop_pos_w)
#   print("rotor_pos_w", rotor_pos_w)
#   print(oMfr.homogeneous)



  # Compute contact point on hoop (lowest point touching ground)
  p_contact = hoop_contact_point(hoop_center, hoop_radius, R_wb)
  rc =  0*hoop_center  # vector from hoop center to contact point
  rc[2] = -0.1  # contact point is on ground plane (z=0)

  rc_dot, rc = get_rc_dot(R_wb, omega, np.array([0.0, 0.0, 1.0]))
#   print("rc_dot", rc_dot)
#   print("rc1", rc1)
#   print("center_vel", Jv @ v)
  # Contact Jacobian: J_c = J_v - skew(r_c) @ J_w
  Jc = Jv - pin.skew(rc) @ Jw

  # Contact velocity
  contact_vel = Jc @ v

  # Contact point height (for detecting ground contact)
#   contact_height = p_contact[2]

  # J_c_dot * v (time derivative of contact Jacobian times velocity)
#   Jc_dot_v = Jdotv.linear + np.cross(Jdotv.angular, rc)

  Jc_dot_v = -np.cross(omega, np.cross(omega, rc)) + np.cross(omega,rc_dot)
#   Jc_dot_v = np.cross(omega,rc_dot)


  # Get mass matrix and bias forces
  M = hoop.data.M.copy()
  nle = hoop.data.nle.copy()  # nle = C(q,v)*v + g(q)

  nv = hoop.model.nv

  # Control inputs (zero for passive simulation)
  tau = np.zeros(nv)
#   print(v)
  # tau[-2] = pendulum torque, tau[-1] = rotor torque (if you want to add control)
#   tau[-2] =  - 0.52*0.03*9.81*np.sin(q[-2])  # simple PD control to swing up pendulum

  # tau[-2] =  -np.sqrt(20)*(theta+np.pi/16) - 0.2*(v[-2]+v[4])   # simple PD control to swing up pendulum
  tau[-2] =  - 0.002*(-v[4] + 3)   # simple PD control to swing up pendulum
  # tau[-2] =  -0.002  # simple PD control to swing up pendulum
  # tau[-2] =  0.2*(v[0]-0.1)  # simple PD control to swing up pendulum
  # if t > 2.0 and t < 2.5:
  #   tau[-1] = -0.0001
  # else:
  #   tau[-1] = 0.0
  # Check if in contact with ground
  if True:  # Contact detected
    # Solve for contact force using constraint dynamics:
    # M @ qddot + nle = tau + Jc^T @ lambda
    # Jc @ qddot = -Jc_dot_v - alpha * contact_vel  (Baumgarte stabilization)
    #
    # From first eq: qddot = M^{-1} @ (tau - nle + Jc^T @ lambda)
    # Substitute into second:
    # Jc @ M^{-1} @ (tau - nle + Jc^T @ lambda) = -Jc_dot_v - alpha * contact_vel
    # Jc @ M^{-1} @ Jc^T @ lambda = -Jc_dot_v - alpha * contact_vel - Jc @ M^{-1} @ (tau - nle)
    #
    # Let A = Jc @ M^{-1} @ Jc^T  (operational space inertia inverse)
    # Let b = -Jc_dot_v - alpha * contact_vel - Jc @ M^{-1} @ (tau - nle)
    # Then: lambda = A^{-1} @ b

    M_inv = np.linalg.inv(M)
    A = Jc @ M_inv @ Jc.T

    # Free acceleration (without contact)
    qddot_free = M_inv @ (tau - nle)

    b = -Jc_dot_v - alpha_baumgarte * contact_vel - Jc @ qddot_free

    # Solve for contact force
    lamda = np.linalg.solve(A, b)

    qddot = qddot_free + M_inv @ Jc.T @ lamda

#     if i % 1000 == 0:
#       print(f"t={i*dt:.3f}s, contact_force: fx={lamda[0]:.3f}, fy={lamda[1]:.3f}, fz={lamda[2]:.3f}")

#   else:
#     # No contact - free fall
#     M_inv = np.linalg.inv(M)
#     qddot = M_inv @ (tau - nle)
#     lamda = np.zeros(3)
#     if i % 1000 == 0:

#       print(f"t={i*dt:.3f}s, no contact, height={contact_height:.4f}")
    # print("a_c",Jc@qddot + Jc_dot_v)
    # print("test_ac", Jc@qddot)
    # print("centripetal", np.cross(omega, np.cross(omega, rc)))
  roll,pitch,yaw = pin.rpy.matrixToRpy(R_wb)
  roll_list.append(roll)
  pitch_list.append(pitch)
  yaw_list.append(yaw)
  t= t+dt
  pend_post = q[-2]
  rotor_post = q[-1]
  # Log data
  times.append(i * dt)
  contact_forces_x.append(lamda[0])
  contact_forces_y.append(lamda[1])
  contact_forces_z.append(lamda[2])
#   contact_heights.append(contact_height)
  vel = J@v
  vel = R_wb @ v[:3]  # Transform linear velocity to body frame

  contact_vel_x.append(vel[0])
  contact_vel_y.append(vel[1])
  contact_vel_z.append(vel[2])
  hoop_pos_x.append(q[0])
  hoop_pos_y.append(q[1])
  hoop_pos_z.append(q[2])
  pend_velocity.append(v[-2]+v[4])
  hoop_velocity.append(v[4])
  pend_pos.append(pend_post)
  rotor_pos.append(rotor_post)
  theta_list.append(theta)
  input_torque_list.append(tau[-2])



  # Semi-implicit Euler integration
  v = v + dt * qddot
  q = pin.integrate(hoop.model, q, dt * v)

  # Update visualization frames and display
  if i % 100 == 0:
    hoop.viz.viewer["rotor_frame"].set_transform(oMfr.homogeneous)
    hoop.viz.viewer["hoop_frame"].set_transform(oMfh.homogeneous)

    trajectory.append(hoop_center + rc)

    # Update trajectory every few steps for speed
    if len(trajectory) > 1 and i % 10 == 0:
        pts = np.array(trajectory).T   # shape (3, N)

        traj_node.set_object(
            mg.Line(
                mg.PointsGeometry(pts),
                mg.LineBasicMaterial(color=0x00ff00)
            )
        )
    hoop.display(q)

# # ============ Plotting ============
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Contact forces
axes[0].plot(times, contact_forces_x, label='fx', color='r')
axes[0].plot(times, contact_forces_y, label='fy', color='g')
axes[0].plot(times, contact_forces_z, label='fz', color='b')
axes[0].set_ylabel('Contact Force [N]')
axes[0].set_title('Contact Forces')
axes[0].legend()
axes[0].grid(True)



# Contact velocity
axes[1].plot(times, contact_vel_x, label='vx', color='r')
axes[1].plot(times, hoop_velocity, label='vy', color='g')
# axes[1].plot(times, contact_vel_z, label='vz', color='b')
axes[1].set_ylabel('Velocity [m/s]')
axes[1].set_title('Contact Point Velocity')
axes[1].legend()
axes[1].grid(True)

# Hoop position
axes[2].plot(times, hoop_pos_x, label='x', color='r')
axes[2].plot(times, hoop_pos_y, label='y', color='g')
axes[2].plot(times, hoop_pos_z, label='z', color='b')
axes[2].set_xlabel('Time [s]')
axes[2].set_ylabel('Position [m]')
axes[2].set_title('Hoop Center Position')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('contact_dynamics.png', dpi=150)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(times, pend_velocity, label='Pendulum Velocity', color='m')
plt.plot(times, hoop_velocity, label='Hoop Velocity', color='c')
plt.xlabel('Time [s]')
plt.ylabel('Angular Velocity [rad/s]')
plt.title('Pendulum and Hoop Velocities')
plt.legend()
plt.grid(True)
plt.savefig('pendulum_hoop_velocities.png', dpi=150)
plt.show()

# plt.figure(figsize=(8, 6))
# plt.plot(times, pend_pos, label='Pendulum Angle', color='m')
# plt.plot(times, rotor_pos, label='Rotor Angle', color='c')
# plt.xlabel('Time [s]')
# plt.ylabel('Angle [rad]')
# plt.title('Pendulum and Rotor Angles')
# plt.legend()
# plt.grid(True)
# plt.savefig('pendulum_rotor_angles.png', dpi=150)
# plt.show()

plt.figure(figsize=(8, 6))
plt.plot(times, theta_list, label='Roll', color='r')
plt.axhline(-np.pi/4, color='k', linestyle='--', label='Target Angle')
plt.xlabel('Time [s]')
plt.ylabel('Theta [rad]')
plt.title('Theta vs Time')
plt.legend()
plt.grid(True)
plt.savefig('theta_vs_time.png', dpi=150)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(times, input_torque_list, label='Input Torque', color='m')
plt.xlabel('Time [s]')
plt.ylabel('Torque [Nm]')
plt.title('Input Torque vs Time')
plt.legend()
plt.grid(True)
plt.savefig('input_torque.png', dpi=150)
plt.show()
# # XY trajectory plot
plt.figure(figsize=(8, 6))
plt.plot(hoop_pos_x, hoop_pos_z, 'b-', label='Trajectory')
plt.scatter(hoop_pos_x[0], hoop_pos_z[0], color='green', s=100, label='Start', zorder=5)
plt.scatter(hoop_pos_x[-1], hoop_pos_z[-1], color='red', s=100, label='End', zorder=5)
plt.xlabel('X position [m]')
plt.ylabel('Y position [m]')
plt.title('Hoop Trajectory (XY plane)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('hoop_trajectory.png', dpi=150)
plt.show()
