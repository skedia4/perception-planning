
import numpy as np
from rotations import skew_symmetric, Quaternion

class ExtendedKalmanFilter:

    def __init__(self, p, v, q, p_cov):


        self.var_imu_f = 0.5**2
        self.var_imu_w = 0.5**2
        self.var_lidar = 5.00

        self.g= np.array([0, 0, -9.81])
        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
        
        self.h_jac = np.zeros([3, 9])
        self.h_jac[:, :3] = np.eye(3) 

        
        self.p_check= p
        self.v_check= v
        self.q_check= q
        self.p_cov_check= p_cov


    def update(self, y_k):

        r_cov = np.eye(3)*self.var_lidar
        
        k_gain = self.p_cov_check @ self.h_jac.T @ np.linalg.inv((self.h_jac @ self.p_cov_check @ self.h_jac.T) + r_cov)
        
        error_state = k_gain @ (y_k - self.p_check)

        self.p_check = self.p_check + error_state[0:3]
        self.v_check = self.v_check + error_state[3:6]
        self.q_check = Quaternion(axis_angle=error_state[6:9]).quat_mult_left(Quaternion(*self.q_check))
        self.p_cov_check = (np.eye(9) - k_gain @ self.h_jac) @ self.p_cov_check


    def predict(self, f, w, dt, q_cov):

       q_prev = Quaternion(*self.q_check)
       q_curr = Quaternion(axis_angle=(w*dt)) 
       c_ns = q_prev.to_mat() 
       f_ns = (c_ns @ f) + self.g 

       self.p_check = self.p_check + dt*self.v_check + 0.5*(dt**2)*f_ns
       self.v_check = self.v_check + dt*f_ns
       self.q_check = q_prev.quat_mult_left(q_curr)

       f_jac = np.eye(9)
       f_jac[0:3, 3:6] = np.eye(3)*dt
       f_jac[3:6, 6:9] = -skew_symmetric(c_ns @ f)*dt

       self.p_cov_check = f_jac @ self.p_cov_check @ f_jac.T + self.l_jac @ q_cov @ self.l_jac.T



