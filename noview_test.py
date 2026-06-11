import time

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from steering_control import WBQP
from robot_model import spinjorr
import pinocchio as pin
import scipy.sparse as sp
import meshcat_shapes
from utils import hoop_contact_point
import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf

path_pts = []


model = mujoco.MjModel.from_xml_path('hoop.xml')
data = mujoco.MjData(model)
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hoop")

mujoco.mj_resetData(model, data)

trajectory = []
times = []
vxs, vys, vzs = [], [], []
forcetorque = np.zeros(6)

accsensordata = []
pend_inp = []
rotor_inp = []
lamda_x = []
lamda_y = []
lamda_z = []
########################################
hoop = spinjorr(visualize=True)
fid_hoop = hoop.model.addFrame(
    pin.Frame(
        "hoop_frame",                 # frame name
        hoop.jid_hoop,          # parent joint
        0,                      # parent frame index (0 = universe, OK here)
        pin.SE3.Identity(),     # placement in parent joint frame
        pin.FrameType.OP_FRAME
    )
)
fid_rotor = hoop.model.addFrame(
    pin.Frame(
        "rotor_frame",                 # frame name
        hoop.jid_rotor,          # parent joint
        0,                      # parent frame index (0 = universe, OK here)
        pin.SE3.Identity(),     # placement in parent joint frame
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

q = hoop.neutral().copy()

theta = np.deg2rad(0.0)
R0 = pin.utils.rotate('z', theta)
R1 = pin.utils.rotate('z', np.deg2rad(0.0))
quat = pin.Quaternion(R1@R0)   # Pinocchio quaternion
#mujoco uses (w,x,y,z) , pinocchio uses (x,y,z,w) for quaternion
qx, qy, qz, qw = quat.coeffs().copy()
data.qpos[3:7] = np.array([qw, qx, qy, qz])
print(data.qpos.copy())

q = data.qpos.copy()
# print(q)
q[3:7] = quat.coeffs()
qd= data.qvel.copy()

dt = 1e-3
T  = 5
N  = int(T / dt)
hoop.data = hoop.model.createData()

for i in range (N):
  vel = data.cvel[body_id][-3:]
  a = data.cacc[body_id][-3:]
  p = data.xpos[body_id]
  pin.forwardKinematics(hoop.model, hoop.data, q, qd,np.zeros(hoop.model.nv))
  # pin.framesForwardKinematics(hoop.model, hoop.data, q)
  pin.updateFramePlacements(hoop.model, hoop.data)
  pin.computeJointJacobians(hoop.model, hoop.data, q)
  pin.computeAllTerms(hoop.model,hoop.data,q,qd)

  fid_rotor = hoop.model.getFrameId("rotor_frame")
  fid_hoop  = hoop.model.getFrameId("hoop_frame")

  oMfr = hoop.data.oMf[fid_rotor]
  oMfh = hoop.data.oMf[fid_hoop]
  # Send to Meshcat
  hoop.viz.viewer["rotor_frame"].set_transform(oMfr.homogeneous)
  hoop.viz.viewer["hoop_frame"].set_transform(oMfh.homogeneous)
  # if data.time <t_roll:
  J = pin.computeFrameJacobian(hoop.model,hoop.data,q,fid_hoop, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
  Jdotv= pin.getFrameClassicalAcceleration(hoop.model,hoop.data,fid_hoop,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
  Jvdot = Jdotv.linear
  J_omegadot = Jdotv.angular
  Jv = J[:3, :]    # linear velocity Jacobian
  Jw = J[3:, :]    # angular velocity Jacobian
  omega = Jw @ qd
  hoop_center  = q[0:3]
  quat = q[3:7]
  R_wb = oMfr.rotation
  # print("hoop_center",hoop_center)
  # print(hoop_contact_point(hoop_center,0.1,R_wb))
  # print(hoop_center)
  # R_wb = R = quat.toRotationMatrix()
  rc = hoop_contact_point(hoop_center,0.1,R_wb) - hoop_center
  Jc = Jv-pin.skew(rc)@Jw
  print("jac",Jc@qd)
  Jc_dot_v =  (Jvdot + np.cross(J_omegadot,rc)) + np.cross(omega, np.cross(omega, rc))
  # print(Jc_dot_v)
  # print("contact_vel",Jc@v)
  # print(p)
  # data.ctrl[1] = 0.0#0.1*(1-v[0]) + 0.01*(-a[0])
  v_d = np.array([0.3,0.4,0])
  wbc  = WBQP(hoop.model,hoop.data,fid_hoop)
  wbc.build_qp(v_d,qd,J , Jdotv,Jc,Jc_dot_v)
  qddot, tau , lamda = wbc.solve(Jc)
  data.ctrl[1] = tau[1]
  data.ctrl[0] = tau[0]
  pend_inp.append(tau[-2])
  rotor_inp.append(tau[-1])
  lamda_x.append(lamda[0])
  lamda_y.append(lamda[1])
  lamda_z.append(lamda[2])
  # data.ctrl[0] = 0.0#0.1*(1-v[0]) + 0.01*(-a[0])
  vel = pin.getFrameVelocity(hoop.model, hoop.data, fid_hoop, pin.ReferenceFrame.LOCAL)

  vel = vel.linear
#   vel = data.qvel[:3].copy()
  vxs.append(vel[0]); vys.append(vel[1]); vzs.append(vel[2])
  times.append(data.time)     # MuJoCo sim time
  ball_pos = p.copy()#data.qpos[:3].copy()
  ball_quat = data.qpos[3:7].copy()
  # print(data.qacc)
  trajectory.append(ball_pos)
  accsensordata.append(data.sensor('accelerometer').data.copy())


  mujoco.mj_step(model, data)
  q = data.qpos.copy()
  qw, qx, qy, qz = q[3:7]
  quat_pin = -np.array([qx, qy, qz, qw])
  q[3:7] = quat_pin

  # qx, qy, qx , qw =data.qpos[3:7]
  qd= data.qvel.copy()



  hoop.display(q)

# --- Plot the trajectory ---
trajectory = np.array(trajectory)  # Convert list to array.
print(trajectory.shape)
plt.figure(figsize=(8, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Ball trajectory")
# plt.plot(times, trajectory[:, 0], label="Ball trajectory")
plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', label='Start')
plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', label='End')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Ball Trajectory in XY plane')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
plt.figure(figsize=(8, 4))
plt.plot(times, vxs, label='vx')
plt.plot(times, vys, label='vy')
plt.plot(times, vzs, label='vz')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Linear Velocity Components vs Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
###
plt.plot(times, lamda_x, label='vx')
plt.plot(times, lamda_y, label='vy')
plt.plot(times, lamda_z, label='vz')
plt.xlabel('Time [s]')
plt.ylabel('lamda')
plt.title('reaction_force')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
#####
plt.plot(times, pend_inp, label='pend_inp')
plt.plot(times, rotor_inp, label='rotor_inp')
plt.xlabel('Time [s]')
plt.ylabel('inp')
# plt.title('Linear Velocity Components vs Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
sensor_data = np.array(accsensordata)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))

ax1.plot(times, sensor_data[:, 0], label='Accelerometer X')
ax1.set_ylabel('Acceleration X [m/s²]')
ax1.grid(True)
ax1.legend()

ax2.plot(times, sensor_data[:, 1], label='Accelerometer Y')
ax2.set_ylabel('Acceleration Y [m/s²]')
ax2.grid(True)
ax2.legend()

# ax3.plot(times, np.sqrt(sensor_data[:, 0]**2 + sensor_data[:, 1]**2), label='Accelerometer Z')
ax3.plot(times, sensor_data[:,2], label='Accelerometer Z')
ax3.set_ylabel('Acceleration Z [m/s²]')
ax3.set_xlabel('Time [s]')
ax3.grid(True)
ax3.legend()

plt.suptitle('Accelerometer Data vs Time')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s²]')
plt.title('Accelerometer Data vs Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()