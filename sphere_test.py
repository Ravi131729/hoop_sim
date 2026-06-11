# import pinocchio as pin
# import numpy as np
# from pinocchio.visualize import MeshcatVisualizer
# import meshcat.geometry as g
# import meshcat.transformations as tf
# import hppfcl
# import time
# m = 0.45       # mass [kg]
# R = 0.1       # radius [m]

# I_scalar = 2/5 * m * R**2
# I = np.diag([I_scalar, I_scalar, I_scalar])

# model = pin.Model()

# jid = model.addJoint(
#     0,                                  # universe
#     pin.JointModelFreeFlyer(),          # 6 DOF
#     pin.SE3.Identity(),
#     "sphere_free_flyer"
# )

# inertia = pin.Inertia(m, np.zeros(3), I)
# model.appendBodyToJoint(jid, inertia, pin.SE3.Identity())

# jid_pend = model.addJoint(
#             jid,
#             pin.JointModelRY(),
#             pin.SE3.Identity(),
#             "pendulum_joint"
#         )
# pendulum_inertia = pin.Inertia(
#             0.45,
#             np.array([0, 0, -0.03]),
#             np.diag([0.00066, 0.00066, 0.00063])
#         )
# model.appendBodyToJoint(
#             jid_pend,
#             pendulum_inertia,
#             pin.SE3.Identity()
#         )

# p_local = np.array([0, 0, 0])
# frame_id = model.addFrame(
#     pin.Frame("sphere_center", jid_pend, 0, pin.SE3(np.eye(3), p_local), pin.FrameType.OP_FRAME)
# )

# data = model.createData()
# q  = pin.neutral(model)

# v= np.zeros(model.nv)
# v[-1] = 1.0     # ω_y
# v[0] = 1.0
# pin.forwardKinematics(model, data, q )
# pin.computeJointJacobians(model,data,q)
# pin.updateFramePlacements(model, data)
# print(frame_id)

# J = pin.computeFrameJacobian(
#     model,
#     data,
#     q,
#     frame_id,
#     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
# )

# Jv = J[:3, :]    # linear velocity Jacobian
# Jw = J[3:, :]    # angular velocity Jacobian


# print(J)
import hppfcl
import numpy as np
import pinocchio as pin
import numpy as np

def hoop_contact_point(c, R, R_wb, n_g=np.array([0.0, 0.0, 1.0]), eps=1e-8):
    """
    Compute closest point on a hoop (circle) to a ground plane.

    Parameters
    ----------
    c : (3,) array
        Hoop center in world frame
    R : float
        Hoop radius
    R_wb : (3,3) array
        Rotation matrix from body to world
    n_g : (3,) array
        Ground normal (unit), default z-up
    eps : float
        Tolerance for degeneracy

    Returns
    -------
    p_contact : (3,) array
        Closest point on hoop to ground
    n_h : (3,) array
        Hoop plane normal (world, fixed sign)
    """

    # --- hoop plane normal in world (body z-axis) ---
    n_h = R_wb @ np.array([0.0, 0.0, 1.0])

    # --- fix normal to point away from ground ---
    if np.dot(n_h, n_g) < 0:
        n_h = -n_h

    # --- project ground normal into hoop plane ---
    n_parallel = n_g - np.dot(n_g, n_h) * n_h
    norm_np = np.linalg.norm(n_parallel)

    # --- degenerate case: hoop plane parallel to ground ---
    if norm_np < eps:
        # choose any consistent in-plane direction (body x-axis)
        d = R_wb @ np.array([1.0, 0.0, 0.0])
        d = d / np.linalg.norm(d)
    else:
        # direction toward ground inside hoop plane
        d = -n_parallel / norm_np

    # --- contact point ---
    p_contact = c + R * d

    return p_contact, n_h

theta = np.deg2rad(90.0)
R0 = pin.utils.rotate('x', theta)
quat = pin.Quaternion(R0)   # Pinocchio quaternion
# Create collision objects
# Ground (as a box or plane)
ground = hppfcl.Box(10.0, 10.0, 0.01)  # Large flat box
ground_transform = hppfcl.Transform3f()
ground_transform.setTranslation(np.array([0, 0, -0.005]))  # Position at z=0

# Cylinder
cylinder_radius =0.1
cylinder_height = 0.1
cylinder = hppfcl.Cylinder(cylinder_radius, cylinder_height)
cylinder_transform = hppfcl.Transform3f()
cylinder_transform.setTranslation(np.array([0, 0, 15]))  # Some position above ground
cylinder_transform.setRotation(R0)
# Create collision request and result
request = hppfcl.DistanceRequest()
request.enable_signed_distance = True
result = hppfcl.DistanceResult()

# Compute distance
distance = hppfcl.distance(
    cylinder, cylinder_transform,
    ground, ground_transform,
    request, result
)
# Use getter methods instead of direct access
closest_point_cylinder = result.getNearestPoint1()
closest_point_ground = result.getNearestPoint2()
c= np.array([0, 0, 15])
R_wb = R0
p, n_h = hoop_contact_point(c, cylinder_radius, R_wb)
print("Contact point:", p)
print("Hoop normal:", n_h)
print("*"*8)
# print(f"Distance: {result.min_distance}")
print(f"Closest point on cylinder: {closest_point_cylinder}")
# print(f"Closest point on ground: {closest_point_ground}")
# # Get closest points
# closest_point_cylinder = result.nearest_points[0]  # On cylinder
# closest_point_ground = result.nearest_points[1]     # On ground
# distance_value = result.min_distance

# print(f"Distance: {distance_value}")
# print(f"Closest point on cylinder: {closest_point_cylinder}")
# print(f"Closest point on ground: {closest_point_ground}")