import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from extended_kalman_filter import  ExtendedKalmanFilter
from rotations import angle_normalize, rpy_jacobian_axis_angle, Quaternion
import pykitti
from seq_data.utils import StampedData
from scipy.spatial.transform import Rotation as R

basedir = 'seq_data/kitti'
date = '2011_09_30'
drive = '0018'

dataset = pykitti.raw(basedir, date, drive)


#   gt: ground truth poses 
#   imu_f: imu specific force data (vehicle frame)
#     data: The actual data
#     t: Time elapsed in seconds.
#   imu_w  imu rotational velocity (vehicle frame).
#     data: The actual data
#     t: Time elapsed in seconds.
#   lidar: poses computed from ICP (velodyne frame).
#     data: The actual data
#     t: Time elapsed in seconds.

gt=np.loadtxt("seq_data/05.txt")

timestamps = np.array(dataset.timestamps)
elapsed = np.array(timestamps) - timestamps[0]
ts = [t.total_seconds() for t in elapsed]

imu_f=StampedData()
imu_f.data= np.array([[o.packet.ax,o.packet.ay,o.packet.az] for o in dataset.oxts])
imu_f.t= np.array(ts)


imu_w=StampedData()
imu_w.data= np.array([[o.packet.wx,o.packet.wy,o.packet.wz] for o in dataset.oxts])
imu_w.t= np.array(ts)


lidar_data=np.loadtxt("seq_data/sequence05.txt")
lidar=StampedData()
lidar.data=lidar_data[:,[3,7,11]]
lidar.t= np.array(ts)



# Transform the LIDAR poses to the IMU frame using KITTI Calib files 
# Calib files give Rotation matrix R_imu2velo and translation vector t_imu2velo.

T_imu2velo = dataset.calib.T_velo_imu
T_velo2imu = np.linalg.inv(T_imu2velo)

R_velo2imu = T_velo2imu[:3,:3] 
t_velo2imu=  T_velo2imu[:3, 3]



# Transform from the LIDAR frame to the IMU frame.
lidar.data = (R_velo2imu @ lidar.data.T).T + t_velo2imu


final_poses=np.zeros([imu_f.data.shape[0], 12]) 

p_est = np.zeros([imu_f.data.shape[0], 3])  
v_est = np.zeros([imu_f.data.shape[0], 3]) 
q_est = np.zeros([imu_f.data.shape[0], 4])  

p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  
q_cov = np.zeros((6, 6)) 

init_p_cov=np.eye(9)*100

# Set initial values.
p_est[0] = np.array([0., 0. , 0.])
v_est[0] = np.array([0., 0. , 0.])
q_est[0] = Quaternion(euler=np.array([0. , 0. , 0.])).to_numpy()
#p_cov[0] = np.zeros(9)  # covariance of estimate
lidar_i = 0
lidar_t = list(lidar.t)

ekf = ExtendedKalmanFilter(p_est[0],v_est[0],q_est[0],init_p_cov)

l=imu_f.data.shape[0]
#l=100
for k in range(1, l): 
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # 1. Update state with IMU inputs
    q_cov[0:3, 0:3] = delta_t**2 * np.eye(3)*ekf.var_imu_f
    q_cov[3:6, 3:6] = delta_t**2 * np.eye(3)*ekf.var_imu_w
    ekf.predict(imu_f.data[k - 1], imu_w.data[k - 1] ,delta_t, q_cov)


    # LIDAR measurements 
    if imu_f.t[k] in lidar_t:
        lidar_i = lidar_t.index(imu_f.t[k])
        ekf.update(lidar.data[lidar_i])
    #ekf.update(lidar.data[k])

    # Update states (save)
    p_est[k, :] = ekf.p_check
    v_est[k, :] = ekf.v_check
    q_est[k, :] = ekf.q_check
    p_cov[k, :, :] = ekf.p_cov_check

final_poses[:,3]= p_est[:,0]
final_poses[:,7]= p_est[:,1]
final_poses[:,11]= p_est[:,2]

p_est_euler = []

for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    r=R.from_euler('xyz',[qc.to_euler()])
    r=r.as_matrix().squeeze(0)
    T=np.column_stack([r,[p_est[i,0],p_est[i,1],p_est[i,2]]])
    final_poses[i]=T.flatten()


np.savetxt("pred.txt",final_poses)
np.save("pred.npy",final_poses)

est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot()
ax.plot(final_poses[:,3], final_poses[:,7], label='Estimated')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Estimated Trajectory')
ax.legend(loc=(0.62,0.77))
plt.show()



