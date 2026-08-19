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
model = mujoco.MjModel.from_xml_path('hoop.xml')
data = mujoco.MjData(model)
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hoop")
rotor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotor")


model.vis.scale.contactwidth = 0.1
model.vis.scale.contactheight = 0.03
model.vis.scale.forcewidth = 0.05
model.vis.map.force = 0.3

# kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "k_at_13_822")
# mujoco.mj_resetDataKeyframe(model, data, kid)
mujoco.mj_resetData(model, data)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
trajectory = []
times = []
vxs, vys, vzs = [], [], []
pvxs, pvys, pvzs = [], [], []
forcetorque = np.zeros(6)
dt = 0
t_roll = 20# Time to apply rolling torque
accsensordata = []
pend_vel = []
hoop_vel= []
lamda_x = []
lamda_y = []
lamda_z = []
pend_angle = []
hoop_angle = []
tau_pend = []
########################################


######################################################
with mujoco.viewer.launch_passive(model, data) as viewer:
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] = True
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
  # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
  # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True
  viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
  viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY


  # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True


  start = time.time()
  while viewer.is_running() and dt <5:
    step_start = time.time()
    dt = time.time() - start
    vel = data.cvel[body_id][-3:]
    a = data.cacc[body_id][-3:]
    p = data.xpos[body_id]
    rotor_pos = data.xpos[rotor_body_id]
    print(f"Time: {dt:.2f} s, Position: {p}")
    print(f"Rotor Position: {rotor_pos}")

    qvel = data.qvel.copy()
    q = data.qpos.copy()

    theta = np.arctan2( p[0]-rotor_pos[0], p[2]-rotor_pos[2])

    data.ctrl[1] = 0.0
    data.ctrl[0] = -np.sqrt(0.4)*(theta+np.pi/6) - 0.2*(qvel[-2]+qvel[4])
    # data.ctrl[0] = 0.01

    tau_pend.append(data.ctrl[0])
    v = data.qvel[:3].copy()
    vxs.append(v[0]); vys.append(v[1]); vzs.append(v[2])
    times.append(data.time)     # MuJoCo sim time
    ball_pos = p.copy()#data.qpos[:3].copy()
    ball_quat = data.qpos[3:7].copy()
    # print(data.qacc)
    trajectory.append(ball_pos)
    accsensordata.append(data.sensor('accelerometer').data.copy())
    pend_angle.append(theta)
    # hoop_angle.append(q[4])

    pend_vel.append(qvel[-2] + qvel[4] ) # Pendulum velocity
    hoop_vel.append(qvel[4] ) # Hoop velocity
    model.light_pos[0][:3] = ball_pos + np.array([0, 0, 10])

    # Update the viewer's camera to follow the ball
    # viewer.cam.lookat[:] = ball_pos  # Center camera on ball
    # viewer.cam.distance = 1
    # viewer.cam.azimuth = 90
    # viewer.cam.elevation = -90
    cam.lookat[:] = ball_pos
    cam.distance =3
    cam.azimuth = 90
    cam.elevation = -50
    # mujoco.mjv_updateCamera(model, data, cam, mujoco.mjtCamera.mjCAMERA_FREE)

    mujoco.mj_step(model, data)

    qd= data.qvel.copy()
    # hoop.display(q)
    viewer.sync()
    # time.sleep(0.0001)

    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

plt.figure()
plt.plot(times, vxs, label='vx')
plt.plot(times, vys, label='vy')
plt.plot(times, vzs, label='vz')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Ball Velocity over Time')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(times, pend_vel, label='pendulum velocity')
plt.plot(times, hoop_vel, label='hoop velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Pendulum and Hoop Velocity over Time')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(times, pend_angle, label='pendulum angle')
# plt.plot(times, hoop_angle, label='hoop angle')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Pendulum and Hoop Angle over Time')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(times, tau_pend, label='Pendulum Torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title('Torque Applied to Pendulum over Time')
plt.legend()
plt.grid()
plt.show()
