import pinocchio as pin
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import meshcat_shapes
class spinjorr:
    def __init__(self, visualize=True):
        urdf_path = "spinjorr.urdf"
        mesh_dir = "."  # folder that contains meshes/barrel_hoop.stl

        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path,
            mesh_dir,
            pin.JointModelFreeFlyer()
        )

        self.data = self.model.createData()
        self.collision_data = pin.GeometryData(self.collision_model)
        self.visual_data = pin.GeometryData(self.visual_model)

        if visualize:
            self._init_visualizer()

    def _init_visualizer(self):
        self.viz = MeshcatVisualizer(
            self.model,
            self.collision_model,
            self.visual_model
        )
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()

        self.viz.displayCollisions(True)
        self.viz.displayVisuals(True)

    def neutral(self):
        return pin.neutral(self.model)

    def display(self, q):
        self.viz.display(q)

# hoop = spinjorr(visualize=True)
# q = pin.neutral(hoop.model)
# v= np.zeros(hoop.model.nv)


# for i, f in enumerate(hoop.model.frames):
#     print(i, f.name, f.parentJoint)
# fid_hoop = hoop.model.addFrame(
#     pin.Frame(
#         "hoop_frame",                 # frame name
#         1,          # parent joint
#         0,                      # parent frame index (0 = universe, OK here)
#         pin.SE3.Identity(),     # placement in parent joint frame
#         pin.FrameType.OP_FRAME
#     )
# )
# meshcat_shapes.frame(
#     hoop.viz.viewer["hoop_frame"],
#     axis_length=0.2,
#     axis_thickness=0.005,
#     opacity=0.8,
#     origin_radius=0.002,
# )

# theta = np.deg2rad(10.0)
# R0 = pin.utils.rotate('x', theta)
# R1 = pin.utils.rotate('y', np.deg2rad(20.0))
# quat = pin.Quaternion(R1@R0)   # Pinocchio quaternion

# q[2] = 0.1
# # q[0] = 1
# q[3:7] = quat.coeffs()
# # q[0:3] = [0.0, 0.0, 0.1]   # move base
# # q[3:7] = [0.0, 0.0, 0.0, 1.0]  # identity quaternion
# q[7] = 0.0  # zero joint angles
# hoop.data = hoop.model.createData()
# # fid_rotor = hoop.model.getFrameId("rotor_frame")
# for i, f in enumerate(hoop.model.frames):
#     print(i, f.name, f.parentJoint)
# pin.forwardKinematics(hoop.model, hoop.data, q, v,np.zeros(hoop.model.nv))
# pin.updateFramePlacements(hoop.model, hoop.data)
# # oMfr = hoop.data.oMf[fid_rotor]   # SE3
# fid_hoop  = hoop.model.getFrameId("hoop_frame")
# oMfh = hoop.data.oMf[fid_hoop]

# # Send to Meshcat
# # hoop.viz.viewer["rotor_frame"].set_transform(oMfr.homogeneous)
# hoop.viz.viewer["hoop_frame"].set_transform(oMfh.homogeneous)
# # or modify it
# while True:
#     hoop.display(q)
#     # print(q)