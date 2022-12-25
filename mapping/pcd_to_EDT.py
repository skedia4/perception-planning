
"""

@author: shubham
"""

import open3d as o3d
import numpy as np
from grid_structure import grid
from time import time
from matplotlib import pyplot as plt
from scipy import ndimage
#from PIL import Image



class pcd_to_EDT(object):
    def __init__(self):
        self.data = np.ones( (700,700), dtype=np.float64)
        self.count=[]
        self.offset_x= 100
        self.offset_y= 100
        self.resolution = 10  ############ 1cm equal 1 pixel
        self.EDT_win = int(500/self.resolution) ####### 500 cm
        self.occ_gid_count = 20
        self.step_size =0.3  ### 1 for all update at once
        self.dynamic_upd = True
        
        
    
    def colorify (self):
        image = np.zeros((self.data.shape[0], self.data.shape[1], 3))
        
        #image[:,:,1] =  1
        image[:,:,0] = 1-self.data
        image[:,:,2] =  self.data -1
        image[:,:,1] =  self.data -0.5
        #image[index_red[:,0],index_red[:,1],:]
        # image = np.where(self.data>0.33 and self.data<0.7, [0,1,0], [1,1,1])
        # image = np.where(self.data>0.7, [0,0,1], [1,1,1])
        
        return image
    
    
    
    def occupancy_grid(self, ar):
        #start_1 = time()
        
        values, counts=grid(ar, False, self.resolution)
        
        
        value_min = np.min(values, axis=0).astype(int)
        
        
        
        if value_min[0] + self.offset_x < 0:
            self.data = np.concatenate(( np.ones((-value_min[0] - self.offset_x+100,self.data.shape[1]), dtype=np.float64), self.data), axis=0)
            self.offset_x = - value_min[0] + 100
        if value_min[1] + self.offset_y < 0:
            self.data = np.concatenate(( np.ones((self.data.shape[0], -value_min[1] - self.offset_y+100), dtype=np.float64), self.data), axis=1)
            self.offset_y = - value_min[1] +100
            
        values+= np.array([self.offset_x,self.offset_y])
        
        value_min+= np.array([self.offset_x,self.offset_y]).astype(int)
        value_max = np.max(values, axis=0).astype(int)
            
        if value_max[0] - self.data.shape[0] >=0 :
            self.data = np.concatenate((self.data, np.ones( (value_max[0] - self.data.shape[0]+100,self.data.shape[1]), dtype=np.float64)), axis=0)

        if value_max[1] - self.data.shape[1] >=0 :
            self.data = np.concatenate((self.data, np.ones( (self.data.shape[0], value_max[1] - self.data.shape[1]+100), dtype=np.float64)), axis=1)

        

        
        occ= np.where(counts<self.occ_gid_count, 1, 0 )
        values = values.astype(int)
        value_x= values[:,0]
        value_y= values[:,1]
        
        
        if self.dynamic_upd is True:
            visible_old = self.data[value_min[0]:value_max[0], value_min[1]:value_max[1]]
            visble_new = np.ones((-value_min[0]+value_max[0], -value_min[1] + value_max[1]))
            visble_new[value_x- value_min[0]-1 , value_y- value_min[1]-1] = occ        
            
            
            self.data[value_min[0]:value_max[0], value_min[1]:value_max[1]]+= self.step_size*(visble_new- visible_old)
      
        else:
            pass
            
            
        
        pass
     
        
    def EDT(self, pos_pixel = None, mul_win=0):
        
        
        
        if pos_pixel is not None:
            x_range= np.array([pos_pixel[0]- self.EDT_win- mul_win, pos_pixel[0] +self.EDT_win + mul_win])
            y_range= np.array([pos_pixel[1]- self.EDT_win- mul_win, pos_pixel[1] +self.EDT_win+ mul_win])
            temp = self.data[x_range[0]:x_range[1], y_range[0]:y_range[1]] 
            temp= np.where(temp>0.8, 1, 0)
            data_dist= ndimage.distance_transform_edt(temp)
            
        else:
            occ = np.where(self.data>0.8, 1, 0)
            occ = occ.astype(np.int16)
            
            data_dist= ndimage.distance_transform_edt(occ)
        
        
        return data_dist


if __name__ == '__main__':
   print("Load a ply point cloud, print it, and render it")
   pcd = o3d.io.read_point_cloud("1596800473208167.pcd")
   
   start_1 = time()
   ar=np.asarray(pcd.points)

   
   cc= pcd_to_EDT()
   start_2 = time()
   
   cc.occupancy_grid(ar)
   start_3 = time()
   data_dist= cc.EDT()
   #plt.imshow(data_dist, interpolation='nearest')
   stop = time()
   plt.imshow(cc.data, cmap='gray',  interpolation='nearest')
   
   plt.show()
   
   # print('Time Grid estimate: ', start_2 - start_1)
   # print('Time occupancy grid: ', start_3 - start_2)
   print('Time EDT: ', stop - start_1)