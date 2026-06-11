import pinocchio as pin
import numpy as np
import hppfcl
from pinocchio.visualize import MeshcatVisualizer
import meshcat

class spinjorr:
    def __init__(self, visualize=True):
        self.model = pin.Model()
        self.geom_model = pin.GeometryModel()

        self._build_model()
        self._build_visuals()

        self.geom_data = pin.GeometryData(self.geom_model)

        if visualize:
            self._init_visualizer()
        self.data = self.model.createData()

    # ==================================================
    # Dynamics model
    # ==================================================
    def _build_model(self):
        # ---------- Hoop (free flyer)
        self.jid_hoop = self.model.addJoint(
            0,
            pin.JointModelFreeFlyer(),
            pin.SE3(np.eye(3), np.array([0, 0, 0])),
            "hoop"
        )

        hoop_inertia = pin.Inertia(
            0.4,
            np.zeros(3),
            np.diag([0.0016, 0.0011, 0.0011])
        )

        self.model.appendBodyToJoint(
            self.jid_hoop,
            hoop_inertia,
            pin.SE3.Identity()
        )

        # ---------- Pendulum
        self.jid_pend = self.model.addJoint(
            self.jid_hoop,
            pin.JointModelRY(),
            pin.SE3.Identity(),
            "pendulum_joint"
        )

        pendulum_inertia = pin.Inertia(
            0.45,
            np.array([0, 0, -0.03]),
            np.diag([0.00066, 0.00066, 0.00063])
        )

        self.model.appendBodyToJoint(
            self.jid_pend,
            pendulum_inertia,
            pin.SE3.Identity()
        )

        # ---------- Rotor
        rotor_offset = pin.SE3(np.eye(3), np.array([0, 0, -0.07]))

        self.jid_rotor = self.model.addJoint(
            self.jid_pend,
            pin.JointModelRZ(),
            rotor_offset,
            "rotor_joint"
        )

        rotor_inertia = pin.Inertia(
            0.03,
            np.zeros(3),
            np.diag([0.000013, 0.000013, 0.000026])
        )

        self.model.appendBodyToJoint(
            self.jid_rotor,
            rotor_inertia,
            pin.SE3.Identity()
        )


    # ==================================================
    # Visual geometry
    # ==================================================
    def _build_visuals(self):
        # ---------- Hoop visual (thin cylinder / disc)
        hoop_shape = hppfcl.Cylinder(0.1, 0.05)

        Rm = pin.utils.rotate('x', np.pi / 2)

        hoop_geom = pin.GeometryObject(
            "hoop_vis",
            self.jid_hoop,        # parent joint
            self.jid_hoop,        # parent frame
            hoop_shape,
            pin.SE3(Rm, np.zeros(3))
        )
        hoop_geom.meshColor = np.array([0.1, 0.1, 0.1, 0.3])
        self.geom_model.addGeometryObject(hoop_geom)

        # ---------- Pendulum visual
        pend_shape = hppfcl.Cylinder(0.01, 0.08)
        pend_placement = pin.SE3(np.eye(3), np.array([0, 0, -0.04]))

        pend_geom = pin.GeometryObject(
            "pendulum_vis",
            self.jid_pend,
            self.jid_pend,
            pend_shape,
            pend_placement
        )
        pend_geom.meshColor = np.array([0.9, 0.1, 0.1, 1.0])
        self.geom_model.addGeometryObject(pend_geom)

        # ---------- Rotor visual
        rotor_shape = hppfcl.Cylinder(0.02, 0.01)

        rotor_geom = pin.GeometryObject(
            "rotor_vis",
            self.jid_rotor,
            self.jid_rotor,
            rotor_shape,
            pin.SE3.Identity()
        )
        rotor_geom.meshColor = np.array([0.1, 0.8, 0.1, 1.0])
        self.geom_model.addGeometryObject(rotor_geom)

    # ==================================================
    # MeshCat
    # ==================================================
    def _init_visualizer(self):
        self.viz = MeshcatVisualizer(
            self.model,
            self.geom_model,
            self.geom_model  # reuse visuals as collision geometry
        )
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
        self.vis =meshcat.Visualizer()

        # self.viz.displayFrames(True)

    # ==================================================
    # Helpers
    # ==================================================
    def neutral(self):
        return pin.neutral(self.model)

    def display(self, q):
        # self.viz.displayFrames(True)
        self.viz.display(q)
