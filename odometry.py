#!/usr/bin/env python3
from re import I
import numpy as np
import open3d as o3d
import cv2
from icp import icp
import copy
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import pykitti


def pts2o3d(xyz):
      
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
      
    return pcd

#dataset    
basedir = 'seq_data/kitti'
date = '2011_09_30'
drive = '0018'

dataset = pykitti.raw(basedir, date, drive)

num_frames=sum(1 for _ in dataset.velo)

#odometry 
prev_scan_pts=pts2o3d(dataset.get_velo(0)[:,:-1])

icp_initial = np.eye(4)
curr_se3 = np.eye(4)
pred_poses=[]

for for_idx, velo in tqdm(enumerate(dataset.velo), total= num_frames, mininterval=5.0 ):
 
    curr_scan_pts = pts2o3d(velo[:,:-1])



    curr_scan_down_pts = curr_scan_pts.voxel_down_sample(voxel_size=0.4)

    prev_scan_down_pts = prev_scan_pts.voxel_down_sample(voxel_size=0.4)



    curr_scan_down_pts = np.asarray(curr_scan_down_pts.points)

    prev_scan_down_pts = np.asarray(prev_scan_down_pts.points)

    odom_transform, _, _ = icp(curr_scan_down_pts, prev_scan_down_pts, init_pose=icp_initial, max_iter=25)

    curr_se3 = np.matmul(curr_se3, odom_transform)
    icp_initial = odom_transform 


    prev_scan_pts = copy.deepcopy(curr_scan_pts)
    pred_poses.append(curr_se3.flatten()[0:12])
    
np.save("seq_data/sequence_05.npy",np.array(pred_poses))
np.savetxt("seq_data/sequence05.txt",np.array(pred_poses))