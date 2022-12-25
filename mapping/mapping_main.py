#!/usr/bin/env python3
from re import I
import numpy as np
import open3d as o3d
import cv2
import copy
import matplotlib.pyplot as plt
import pykitti
#import UtilsPointcloud as Ptutils
import argparse
import os
from tqdm import tqdm

#mapping 
from pcd_to_EDT import pcd_to_EDT
cc= pcd_to_EDT()

# dataset
################################ change the directory structure to match point cloud data

basedir = '../seq_data/kitti'
date = '2011_09_30'
drive = '0018'

dataset = pykitti.raw(basedir, date, drive)

num_frames=sum(1 for _ in dataset.velo)

#odometry 
pcd_map = np.empty(shape=[0, 3])
poses = np.load ('pred.npy')

for for_idx, velo in tqdm(enumerate(dataset.velo), total=num_frames, mininterval=5.0):

    
    if for_idx>2000:
        break
    if for_idx<0:
        continue

    curr_scan_pts = velo[:,:-1]

    curr_se3 = poses[for_idx]
    curr_se3 = curr_se3.reshape(3,4)
    curr_se3 = np.vstack((curr_se3.reshape(3,4), np.array([0,0,0,1])))

    prev_scan_pts = copy.deepcopy(curr_scan_pts)
    curr_scan_pts = np.hstack((curr_scan_pts, np.ones((curr_scan_pts.shape[0],1)))).T
    pcd_map = np.concatenate((pcd_map, (curr_se3@ curr_scan_pts)[0:3].T))
    
    if for_idx% 100 ==0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_map ) 
        pcd= pcd.voxel_down_sample(voxel_size=0.07)#0.07
        cc.occupancy_grid(np.asarray(pcd.points))
        pcd_map = np.empty(shape=[0, 3])

    
# %%
#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(pcd_map ) 
#pcd= pcd.voxel_down_sample(voxel_size=0.07)   
#mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
#o3d.visualization.draw_geometries([pcd,mesh_frame])


#cc.occupancy_grid(np.asarray(pcd.points))
#data_dist= cc.EDT()

# %%
from scipy import ndimage, misc

print("till here_1")

img = cc.data.astype(np.int16)

print("till here_2")
result = ndimage.median_filter(img, size=5).astype(np.int16)
print("till here_3")
data_dist= ndimage.distance_transform_edt(result)
fig, ax = plt.subplots(1,2)
ax[0].imshow(result, cmap ="gray")
ax[1].imshow(img)
#plt.show()

outfile = "seq_005_map_data_till_0-400"

fig = plt.figure()
plt.imshow(result, cmap ="gray")
plt.axis('off')
fig.savefig('map_seq_05_zoom.png', dpi=400, bbox_inches='tight')



#np.savez(outfile, np.asarray(pcd.points),  result, data_dist, cc.offset_x, cc.offset_y )

