import pinocchio as pin
import numpy as np
import scipy.sparse as sp
import osqp



class WBQP:
    def __init__(self, model, data, hoop_frame_id):
        self.model = model
        self.data  = data
        self.nv    = model.nv
        self.n_slack = 3
        self.na = 2
        self.fid   = hoop_frame_id

        self.k_r = 10
        self.k_w = 5

        self.P_task = np.zeros((6, 6))
        self.P_task[0:3, 0:3] = np.eye(3)*1
        # print(self.P_task)
        # torque regularization weight
        self.w_slack = 1
        self.w_qdd = 1e-6
        self.w_lam = 1e-5
        self.alpha = 5

        self.prob = None

    def PD_alpha_des(self, v_d):
        # Use frame placement for hoop frame
        oMf = self.data.oMf[self.fid]
        R   = oMf.rotation

        # get angular velocity in WORLD frame (consistent)
        vel = pin.getFrameVelocity(self.model, self.data, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        e_v = v_d - vel.linear

        # Compute desired yaw angle from velocity direction
        # to steer the hoop towards the desired velocity direction
        v_d_norm = np.linalg.norm(v_d[:2])
        if v_d_norm > 0.01:
            desired_yaw = np.arctan2(v_d[1], v_d[0])
            # Current hoop yaw (from rotation matrix)
            current_yaw = np.arctan2(R[1, 0], R[0, 0])
            yaw_error = desired_yaw - current_yaw
            # Wrap to [-pi, pi]
            yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
            print(f"yaw_error: {np.rad2deg(yaw_error):.1f} deg, desired_yaw: {np.rad2deg(desired_yaw):.1f} deg")
        else:
            yaw_error = 0.0

        a_des = -self.k_w * e_v
        print(a_des, e_v)
        print("vel+act", vel.linear)
        return a_des, yaw_error

    def build_qp(self, v_d,qd,J , Jdotv,Jc,Jc_dot_v):
        nv = self.nv
        nc= 3
        ns=self.n_slack
        na =self.na

        # Make sure data.M and data.nle are up to date (call computeAllTerms before)
        M = self.data.M.copy()
        b = self.data.nle.copy()



        a_des, yaw_error = self.PD_alpha_des(v_d)

        # Store yaw_error to apply steering torque later
        self.yaw_error = yaw_error

        xddot_des =np.zeros(6)
        xddot_des[0:3] = a_des

        c = (Jdotv - xddot_des)  # (6,)
        # print(Jdotv)

        # ----- COST for x = [qddot; tau] -----
        H_qdd = J.T @ self.P_task @ J + self.w_qdd * np.eye(nv)
        g_qdd = J.T @ self.P_task @ c
        # print(H_qdd)
        # add torque cost (optional): 0.5 * w_tau * ||tau||^2
        P = sp.block_diag(
            [sp.csc_matrix(H_qdd), sp.eye(nc, format="csc") * self.w_lam, sp.eye(ns, format="csc") * self.w_slack ],
            format="csc"
        )

        q = np.zeros(nv+nc+ns)
        q[:nv] = g_qdd   # correct sign for OSQP form 0.5 x^T P x + q^T x

        # ----- CONSTRAINTS -----
        # dynamics: M qddot - tau = -b
        # A_dyn = sp.hstack([sp.csc_matrix(M), -sp.eye(nv, format="csc") , Jc.T], format="csc")
        # l_dyn = -b
        # u_dyn = -b

        #floating dynamics(equality constraint)
        nf = self.nv-self.na
        M_f = M[0:nf,:]
        b_f = b[0:nf]
        Jcf = Jc[:,0:nf]
        Af_dyn = np.hstack([
            M_f,                 # qddot
            -Jcf.T,              # lambda
            np.zeros((nf,ns))    # slack variables
        ])

        lf_dyn = -b_f
        uf_dyn = -b_f

        #no slip with baugamerte stabilisation
        A_noslip = np.hstack([Jc , np.zeros((nc,nc)),-np.eye(nc)])
        contact_vel = Jc@qd

        # If contact velocity in z is positive (lifting), relax the z constraint
        # to prevent infeasibility
        l_noslip = -Jc_dot_v - self.alpha*contact_vel
        u_noslip = -Jc_dot_v - self.alpha*contact_vel

        # Allow z-direction to be an inequality if lifting (contact_vel[2] > threshold)
        if contact_vel[2] > 0.01:  # hoop is lifting
            # Don't try to pull it down, just constrain xy
            l_noslip[2] = -np.inf  # allow any z acceleration
            u_noslip[2] = np.inf

        #torque limits
        M_a = M[nf:,:]
        b_a = b[nf:]
        Jca = Jc[:,nf:]

        A_torque = np.hstack([M_a, -Jca.T, np.zeros((na,ns))])
        tau_min= np.array([-0.05, -0.1])
        tau_max = np.array([ 0.05, 0.1])
        tau_u = tau_max - b_a
        tau_l = tau_min - b_a

        #friction limits
        mu = 0.8

        n_vars = nv+nc+ns          # total decision variables
        lambda_start = nv         # start index of lambda
        A_f = np.array([
            [ 0.0,  0.0, -1.0],        # -fz <= 0  -> fz >= 0
            [ 1.0,  0.0, -mu ],        #  fx - mu*fz <= 0
            [-1.0,  0.0, -mu ],        # -fx - mu*fz <= 0
            [ 0.0,  1.0, -mu ],        #  fy - mu*fz <= 0
            [ 0.0, -1.0, -mu ],        # -fy - mu*fz <= 0
        ])
        A_friction = np.zeros((5, n_vars))
        A_friction[:, lambda_start:lambda_start+nc] = A_f
        # print(A_friction)
        l_friction = -np.inf * np.ones(5)
        # Allow small violation of fz >= 0 to avoid infeasibility when contact is marginal
        u_friction = np.array([0.1, 0.0, 0.0, 0.0, 0.0])  # small slack on normal force constraint

        #slack limits
        A_slack = np.hstack([np.zeros((ns,nv)),np.zeros((ns,nc)),np.eye(ns)])
        u_slack =  150*np.ones(ns)
        l_slack = -150*np.ones(ns)

        # Explicit bounds on lambda to prevent unboundedness
        A_lambda = np.hstack([np.zeros((nc, nv)), np.eye(nc), np.zeros((nc, ns))])
        l_lambda = np.array([-50.0, -50.0, 0.0])   # fx, fy can be negative; fz >= 0
        u_lambda = np.array([50.0, 50.0, 100.0])   # upper bounds on contact forces
        ################

        A = sp.vstack([Af_dyn, A_torque,A_noslip,A_friction,A_slack, A_lambda], format="csc")
        l = np.concatenate([lf_dyn, tau_l ,l_noslip ,l_friction,l_slack, l_lambda])
        u = np.concatenate([uf_dyn, tau_u, u_noslip , u_friction,u_slack, u_lambda])

        self.prob = osqp.OSQP()
        # self.prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=True, warm_start=True)
        self.prob.setup(
                P=P, q=q, A=A, l=l, u=u,
                verbose=False, warm_start=False,
                polish=False,
                eps_abs=1e-5, eps_rel=1e-5,
                max_iter=20000
            )

    def solve(self,Jc):
        res = self.prob.solve()
        if res.info.status_val not in (1, 2):
            print(f"QP failed with status: {res.info.status}")
            print(f"Returning zero torques")
            # Return safe fallback values instead of crashing
            qddot = np.zeros(self.nv)
            lamda = np.array([0.0, 0.0, 1.0])  # small normal force
            tau = np.zeros(self.na)
            return qddot, tau, lamda
        x = res.x
        qddot = x[:self.nv]
        lamda   = x[self.nv:self.nv+3]
        M = self.data.M.copy()
        b = self.data.nle.copy()
        nf = self.nv-self.na
        M_a = M[nf:,:]
        b_a = b[nf:]
        Jca = Jc[:,nf:]
        tau = M_a@qddot + b_a - Jca.T@lamda

        # Add steering torque to rotor (tau[1]) based on yaw error
        # This helps the hoop turn towards the desired velocity direction
        # Negative sign because positive rotor torque should reduce positive yaw error
        k_yaw = 0.08  # steering gain
        steering_torque = -k_yaw * self.yaw_error
        tau[1] = tau[1] + np.clip(steering_torque, -0.1, 0.1)

        return qddot, tau , lamda