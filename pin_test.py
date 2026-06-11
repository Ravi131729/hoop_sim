import pinocchio as pin
import numpy as np
from robot_model import spinjorr
from utils import hoop_contact_point

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
p_local = np.array([0, 0, 0.0])
contact_fid= hoop.model.addFrame(
    pin.Frame(
        "contact_frame",                 # frame name
        hoop.jid_hoop,          # parent joint
        0,                      # parent frame index (0 = universe, OK here)
        pin.SE3(np.eye(3), p_local),     # placement in parent joint frame
        pin.FrameType.OP_FRAME
    )
)
# print(fid_hoop)
# # #############################
dt = 1e-4
T  = 5
N  = int(T / dt)

q = hoop.neutral().copy()
q[2] = 0.1

# 30 degrees about Y
theta = np.deg2rad(0.0)
R0 = pin.utils.rotate('z', theta)
R1 = pin.utils.rotate('z', np.deg2rad(0.0))
quat = pin.Quaternion(R1@R0)   # Pinocchio quaternion
print(quat.coeffs())
# Set quaternion into q
q[3:7] = quat.coeffs()

v = np.zeros(hoop.model.nv)


hoop.model.gravity = pin.Motion.Zero()
hoop.data = hoop.model.createData()

# # ---- TIME LOOP ----
for i in range(N):

    pin.forwardKinematics(hoop.model, hoop.data, q, v,np.zeros(hoop.model.nv))
    pin.updateFramePlacements(hoop.model, hoop.data)
    pin.computeJointJacobians(hoop.model, hoop.data, q)
    pin.computeAllTerms(hoop.model,hoop.data,q,v)



    qdd = pin.aba(hoop.model,hoop.data,q,v,tau)
    v = v + dt * qdd
    q = pin.integrate(hoop.model, q, dt * v)

    print(q)
    hoop.display(q)















