import pinocchio as pin
import numpy as np
import scipy.sparse as sp
import osqp



class WBQP:
    def __init__(self, model, data, hoop_frame_id,lam=3e-4, w_ang=100.0, w_tau=1e-4):
        self.model = model
        self.data  = data
        self.nv    = model.nv
        self.fid   = hoop_frame_id

        self.k_r = 400
        self.k_w = 32
        self.lam = lam

        # task weight: only angular rows of a 6D spatial accel
        self.P_task = np.zeros((6, 6))
        self.P_task[3:6, 3:6] = np.eye(3) * w_ang
        # print(self.P_task)
        # torque regularization weight
        self.w_tau = 0*0.01

        self.prob = None

    def PD_alpha_des(self, R_d, w_d):
        # Use frame placement for hoop frame
        oMf = self.data.oMf[self.fid]
        R   = oMf.rotation

        # get angular velocity in WORLD frame (consistent)
        vel = pin.getFrameVelocity(self.model, self.data, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # print(vel)
        w   = vel.angular
        R_err = R @ R_d.T          # world-frame error
        e_R = pin.log3(R_err)
        # e_R = pin.log3(R_d.T @ R)
        # e_full = pin.log3(R_d.T @ R)
        # e_R = np.array([0.0, e_full[1], 0.0])

        e_w = w - w_d
        alpha_des =  -self.k_r * e_R - self.k_w * e_w
        # print(alpha_des,e_R)
        return R@alpha_des

    def build_qp(self, R_d, w_d,J , Jdotv):
        nv = self.nv

        # Make sure data.M and data.nle are up to date (call computeAllTerms before)
        M = self.data.M.copy()
        b = self.data.nle.copy()



        alpha_des = self.PD_alpha_des(R_d, w_d)

        xddot_des = np.zeros(6)
        xddot_des[3:6] = alpha_des

        c = (Jdotv - xddot_des)  # (6,)
        # print(Jdotv)

        # ----- COST for x = [qddot; tau] -----
        H_qdd = J.T @ self.P_task @ J + self.lam * np.eye(nv)
        g_qdd = J.T @ self.P_task @ c
        # print(H_qdd)
        # add torque cost (optional): 0.5 * w_tau * ||tau||^2
        P = sp.block_diag(
            [sp.csc_matrix(H_qdd), sp.eye(nv, format="csc") * self.w_tau],
            format="csc"
        )

        q = np.zeros(2 * nv)
        q[:nv] = g_qdd   # correct sign for OSQP form 0.5 x^T P x + q^T x

        # ----- CONSTRAINTS -----
        # dynamics: M qddot - tau = -b
        A_dyn = sp.hstack([sp.csc_matrix(M), -sp.eye(nv, format="csc")], format="csc")
        l_dyn = -b
        u_dyn = -b
        tau_min = -np.zeros(nv)
        tau_max = np.zeros(nv)
        # tau_max[0:3] = np.zeros(3)
        # tau_min[0:3] = np.zeros(3)
        tau_min[-2] =-5
        tau_max[-2] =  5

        tau_min[-1] = -10
        tau_max[-1] =  10

        # torque limits: tau_min <= tau <= tau_max
        A_lim = sp.hstack([sp.csc_matrix((nv, nv)), sp.eye(nv, format="csc")], format="csc")
        l_lim = tau_min
        u_lim = tau_max

        A = sp.vstack([A_dyn, A_lim], format="csc")
        l = np.concatenate([l_dyn, l_lim])
        u = np.concatenate([u_dyn, u_lim])

        self.prob = osqp.OSQP()
        # self.prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, warm_start=True)
        self.prob.setup(
                        P=P, q=q, A=A, l=l, u=u,
                        verbose=True, warm_start=True,
                        polish=True,
                        eps_abs=1e-5, eps_rel=1e-5,
                        max_iter=20000
                    )


    def solve(self):
        res = self.prob.solve()
        if res.info.status_val not in (1, 2):
            raise RuntimeError(res.info.status)
        x = res.x
        qddot = x[:self.nv]
        tau   = x[self.nv:]
        return qddot, tau