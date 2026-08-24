from quantities import hoop_angular_momentum_com,angular_momentum,hoop_position_relative_to_com,center_of_mass_velocity
import matplotlib.pyplot as plt
import numpy as np
from parameters import ModelParams

from dynamics import build_dynamics, impact_map


p_val = ModelParams().as_array()
x0 = np.array([0.0, 0.3, 0.0, 0.0,
               1.0, 0.0, 10.0, 0.0])
MAX_SPEED = 200
r_h_com = hoop_position_relative_to_com(x0, p_val)
l_com = np.linalg.norm(r_h_com)

def get_com_vel(theta,y0,dx0,dy0,params):
    R = params[5]
    g = params[4]
    return np.array([dx0,np.sqrt(2*g*(y0-(R-l_com*np.cos(theta))) )])



def get_hoop_vel(v_com,theta,theta_dot,params):

    return v_com + np.array([l_com*np.cos(theta)*theta_dot,-l_com*np.sin(theta)*theta_dot] )


def get_pendulum_vel(v_com,theta,theta_dot,params):
    hoop_vel = get_hoop_vel(v_com,theta,theta_dot,params)
    return hoop_vel + np.array([-params[3]*np.cos(theta)*theta_dot,params[3]*np.sin(theta)*theta_dot] )


def hoop_angular_momentum_com(x, p):
    Io = p[0]
    mo = p[2]

    dphi = x[6]

    r = hoop_position_relative_to_com(x, p)
    v = hoop_velocity_relative_to_com(x, p)

    H_hoop_com = (
        Io * dphi

        - mo * (
            r[0] * v[1]
            - r[1] * v[0]
        )
    )

    return H_hoop_com







def get_angular_momentum(theta,theta_dot,params):
    mp = params[1]
    lp = params[3]

    hoop_pos_wrt_com =l_com*np.array([np.sin(theta),np.cos(theta)])
    pend_pos_wrt_com = (lp-l_com)*np.array([-np.sin(theta),-np.cos(theta)])
    pend_vel_wrt_com = (lp-l_com)*np.array([-np.cos(theta)*theta_dot,np.sin(theta)*theta_dot])
    hoop_vel_wrt_com = l_com*np.array([np.cos(theta)*theta_dot,-np.sin(theta)*theta_dot])

    H_pend_com = -mp * (
        pend_pos_wrt_com[0] * pend_vel_wrt_com[1]
        - pend_pos_wrt_com[1] * pend_vel_wrt_com[0]
    )

    H_hoop_com = -mp * (
        hoop_pos_wrt_com[0] * hoop_vel_wrt_com[1]
        - hoop_pos_wrt_com[1] * hoop_vel_wrt_com[0]
    ) #with no i phi_dot

    return H_hoop_com , H_pend_com




#initial angular momentum

h0 = angular_momentum(x0,p_val)



theta_grid = np.linspace(0,2*np.pi,200)
theta_dot_grid = np.linspace(-200,200,200)


feasible_points = []
int_points = []
for theta in theta_grid:
    for theta_dot in theta_dot_grid:
        H_hoop_com, H_pend_com = get_angular_momentum(theta,theta_dot,p_val)

        I = p_val[0]

        phi_dot = (h0-H_hoop_com - H_pend_com)/I

        v_com = get_com_vel(theta,x0[1],x0[4],x0[5],p_val)

        hoop_vel = get_hoop_vel(v_com,theta,phi_dot,p_val)

        X = [0.0, 0.3, 0.0, theta,
                      hoop_vel[0], hoop_vel[1], phi_dot, theta_dot]



        qdot_plus, lambda_t, lambda_n, mode = impact_map(
            X[4:8],
            X[0:4],
            p_val,

        )

        X_plus = X.copy()
        X_plus[4:8] = qdot_plus

        v_com_plus = center_of_mass_velocity(X_plus, p_val)

        alpha = np.arctan2(v_com_plus[1],v_com_plus[0])


        omega_vel = theta_dot - phi_dot

        if abs(float(omega_vel)) <= MAX_SPEED and alpha > np.pi/3 and alpha < np.pi/2.8 and v_com_plus[1] >4.0 and v_com_plus[0] >0.0:

            feasible_points.append(X)

        if abs(float(omega_vel)) <= MAX_SPEED :

            int_points.append(X)



points = np.array(feasible_points)
points2 = np.array(int_points)


# points2 = np.array(infeasible_points)
plt.scatter(points2[:, 3], points2[:, 7],  s=5,label='Feasible Points')
plt.scatter(points[:, 3], points[:, 7], s=5,label='Feasible Points with COM Velocity > 3 m/s')

plt.xlabel(r'$\theta$')
plt.ylabel(r'$\dot{\theta}$')
plt.legend()
plt.show()




# vx=[]
# vy=[]
# thetal=np.linspace(0,2*np.pi,100)
# for theta in np.linspace(0,2*np.pi,100):
#     dx,dy = get_com_vel(theta,x0[1],x0[4],x0[5],p_val)
#     vx.append(dx)
#     vy.append(dy)

# plt.plot(thetal,vy)
# plt.show()