#!/usr/bin/env python3
# -*-coding:utf-8 -*-
'''
@Time    :   2022/05/06 18:14:30
@Author  :   Yu Zhou 
@Contact :   yuzhou7@illinois.edu
'''

from matplotlib.pyplot import sca
from planning.misc import identity
import numpy as np
import yaml

class LoadMap:
    def __init__(self, path) -> None:
        self.data = self.load(path)
    
    def load(self, path):
        with open(path) as configFile:
            mapDict = yaml.safe_load(configFile)
            map_log = np.load(mapDict['map_path'], allow_pickle = True)
            cc, edt_img = map_log[mapDict['occupancy_img']], map_log[mapDict['edt_img']]
            off_h, off_w, scale = mapDict['offset_h'], mapDict['offset_w'], mapDict['scale']
            edt = EDTmap.fromImage(edt_img/scale, resolution = 1/scale,  off_h = off_h, off_w = off_w,flip_y=False) #from img to world
        return cc, edt_img, off_h, off_w, scale, edt
class EDTmap:
    """
    Treats a 2D image as a EDTmap.
   
    In local coordinates, the EDTmap is centered at (0,0) and has width
    (x-dimension) dimensions[0] and height (y-dimension) dimensions[1].
   
    Attributes:
        frame (Klampt se3 object): the reference frame of the center of the
            EDTmap
        dimensions (pair of floats): the map size in the local x and y
            directions
        edt_grid (np.ndarray): the grid of EDTs (Z direction).  The
            index i,j indicates x (low to high), y (low to high).
        color_grid (np.ndarray, optional): the grid of RGB colors.  Must have
            shape (w,h,3) with edt_grid having the shape (w,h).  If the
            dtype is uint8, the range is assumed to be [0,255].  Otherwise, it
            is assumed to be [0,1].
       
    Note that the EDTmap is interpreted to take up the full dimensions of
    the rectangle, i.e., are defined at grid vertices, not cell centers.
    """
    def __init__(self,frame,dimensions,edt_grid, off_h = 0.5, off_w = 0.5):
        self.frame = frame
        self.dimensions = dimensions
        self.edt_grid = edt_grid
        self.color_grid = None
        self.off_h = off_h 
        self.off_w = off_w 

    def index(self,pt,local=True,clamp=False):
        """Converts from a point to a grid index.
       
        If the coordinates of the point are outside of the range, and clamp
        =True, then the closest valid grid index will be returned.  Otherwise,
        an index less than 0 or greater than width-1/height-1 may be returned.
       
        Returns:
            tuple: ((i,j),(u,v)) where (i,j) are the indices of the lower-left
            corner of the grid cell in which pt is located, and (u,v) are the
            coordinates within the cell, both in the range [0,1].
        """
        # if self.edt_grid is None: return None
        # if not local and self.frame is not None:
            # pt = se3.apply(se3.inv(self.frame),pt)
        u = (self.edt_grid.shape[0]-1)*(pt[0]/self.dimensions[0] + self.off_h) # be caucious about the coordinate of edt: +0.5 if it's from the center
        v = (self.edt_grid.shape[1]-1)*(pt[1]/self.dimensions[1] + self.off_w)
        i,j = int(u),int(v)
        # print("pt {} {} uv {} {} ij {} {}".format(pt[0],pt[1],u,v,i,j))
        if clamp:
            if i < 0:
                i,u=0,0
            elif i >= self.edt_grid.shape[0]:
                i,u=self.edt_grid.shape[0]-1,self.edt_grid.shape[0]
            if j < 0:
                j,v=0,0
            elif j >= self.edt_grid.shape[1]:
                j,v=self.edt_grid.shape[1]-1,self.edt_grid.shape[1]
        return (i,j),(u-i,v-j)
   
    def edt_dist(self,pt,local=True):
        """Reads the EDT of a point.
       
        Args:
            pt (list or tuple): a 2D or 3D point
            local (bool): whether the point is in local coordinates or world
                coordinates.
            interpolation (str): either 'nearest' or 'bilinear' describing how
                the interpolation is done between nearby EDTmap values.
        """
        (i,j),(u,v) = self.index(pt,local=local)
        return bilinear_interpolate(self.edt_grid,i,j,u,v)
   
    @staticmethod
    def fromImage(fn_or_img,width=None,height=None,resolution=0.1,
        zscale= 1.0,zmax=1.0,flip_y=False, off_h = None, off_w = None):
        """Converts an image to a EDTmap.

        If width and/or height are given, the EDTmap is scaled to have that
        overall width/height.  If resolution is given, the EDTmap is scaled
        so that each pixel has that width and height.
       
        Args:
            fn_or_img (str, PIL Image, or numpy array): the input
            width (float or None): width (X dimension) of EDTmap
            height (float or None): height (Y dimension) of EDTmap
            resolution (float or None): pixel size of EDTmap.
            zscale (float or 'auto', optional): if 'auto', rescales the
                EDTmap to the range [zmin,zmax]
            zmax (float or None): the z range maximum, used if zscale='auto'.
                If zscale='auto' and zmax is None, no scaling is performed.
            flip_y (bool): whether to treat the [0,0] coordinate
                as the upper left of the image (should be true for PIL images)
            color (str, PIL, or numpy array, optional): the colors.  Must be the
                same width and height of the depth image.
        """
        if isinstance(fn_or_img,str):
            fn = fn_or_img
            try:
                from PIL import Image
            except ImportError:
                import Image
            img = Image.open(fn)
            img_array = np.asarray(img)
            if len(img_array.shape) == 3:
                img = img_array.dot([1.0/3]*3)
            else:
                img = img_array
        else:
            img = np.asarray(fn_or_img)
        if len(img.shape) != 2:
            raise ValueError("Invalid image shape")
        # print("Image shape",img.shape)
        if zscale == 'auto':
            if zmax is None:
                zscale = 1.0
            else:
                zscale = zmax / img.max()
                print("Image max",img.max())
        print("zscale:",zscale)
        if height is None:
            if width is None:
                if resolution is not None:
                    width = max(1,img.shape[1]-1)*resolution
                    height = max(1,img.shape[0]-1)*resolution
                    print(f"Map width: {width} and heigt: {height}")
                else:
                    width = 1.0
            if height is None:
                height = width*img.shape[0]/img.shape[1]
        elif width is None:
            width = height*img.shape[1]/img.shape[0]
        print(off_h, off_w)
        if off_h is None:
            off_h = 0.5
        else: 
            off_h = round(off_h/img.shape[0], 2)
        if off_w is None:
            off_w = 0.5
        else: 
            off_w = round(off_w/img.shape[1], 2) 

        if flip_y:
            res = EDTmap(identity(),(height, width),img[::-1,:]*zscale, off_h=off_h, off_w=off_w)
        else:
            res = EDTmap(identity(),(height, width),img*zscale, off_h=off_h, off_w=off_w)
        return res
    
    def calcDistCost(self, state, thresh = 0.5):
        dist, grad = self.edt_dist(state)
        if dist == None: return 10000
        a, b = 4, 2
        cost = np.exp(-a * (dist - 0.8)) * b # works better than the discrete one
        # if dist <= thresh:
        #     # cost = 1/(dist)**2
        #     # cost = -np.log(dist/thresh)
        #     cost = (dist - thresh)**2 
        #     # cost = (1/dist - 1/thresh)**2/100
        # else:
        #     cost = 0
        return cost

def bilinear_interpolate(grid,i,j,u,v,res_inv = 10):
    if i < 0 or i+1 >= grid.shape[0]:
        return None, None
    if j < 0 or j+1 >= grid.shape[1]:
        return None, None
    # value = (1-u) * (1-v) * grid[i,j] + u * v * grid[i+1,j+1] + u * (1 - v) * grid[i+1,j] + v * (1 - u) * grid[i,j+1] # takes more time this way
    w0 = grid[i,j] + u*(grid[i+1,j]-grid[i,j])
    w1 = grid[i,j+1] + u*(grid[i+1,j+1]-grid[i,j+1])
    value = w0 + v*(w1-w0)
    # grad = np.zeros(2)
    # grad[0] = ((1 - v) * (grid[i+1,j] - grid[i,j]) + v * (grid[i+1,j+1] - grid[i,j+1])) * res_inv
    # grad[1] = (w1 - w0) * res_inv
    grad = []
    return value, grad