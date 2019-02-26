#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 02:33:47 2019

@author: jdeguzman
"""
import numpy as np
import cv2
import os
import pickle
import time
import matplotlib.pyplot as plt
from PIL import Image
from slam_functions import body2world

def load_images_from_folder(path):
    images = []
#    for filename in os.listdir(path):
#        images.append(filename)
#        prefix, num = filename[:-4].split('_')
#        num = num.zfill(4)
#        new_filename = prefix + "_" + num + ".png"
#        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
    
    for filename in os.listdir(path):
        images.append(filename)
    return sorted(images)

folder = '/Users/jdeguzman/Desktop/ECE276A_HW2/dataRGBD/Disparity21'
folder1 = '/Users/jdeguzman/Desktop/ECE276A_HW2/dataRGBD/RGB21'
filenames = load_images_from_folder(folder)
filenames1 = load_images_from_folder(folder1)
#
#fname = open('/Users/jdeguzman/Desktop/ECE276A_HW2/data/DR21.pkl', 'rb')
#trajectory = pickle.load(fname)
#fname.close()

#%%

def homogenize(x):
    # converts points from inhomogeneous to homogeneous coordinates
    return np.vstack((x,np.ones((1,x.shape[1]))))

def dehomogenize(x):
    # converts points from homogeneous to inhomogeneous coordinates   
    return x[:-1]/x[-1]

def projection(X): 
    # input scene points need to be 4xN matrix
    # returns 3xN homogeneous image points
    
    # calibration matrix w/ intrinsic parameters
    K = np.array([[585.05108211, 0,            242.94140713],
                  [0,            585.05108211, 315.83800193],
                  [0,            0,                       1]])
    P = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]])
    Z = X[2,:]
    x = K @ P @ X
    return x/Z

def optical2body(T_oc, T_bc):
    T_bo = T_bc @ np.transpose(T_oc)
    return T_bo 

def get_RGBD(disp):
    d = disp
    m, n = d.shape[0], d.shape[1]
    d = d.reshape(m*n,1)
    dd = -0.00304*d + 3.31
    depth = 1.03 / dd
    c, r = np.meshgrid(np.arange(m),np.arange(n))
    c = c.reshape(m*n,1)
    r = r.reshape(m*n,1)
    rgb_i = ((c * 526.37 + dd*(-4.5 * 1750.46) + 19276) / 585.051).astype(int)
    rgb_j = ((r * 526.37 + 16662) / 585.051).astype(int)    
    indValid = depth >= 0
    depth = depth[indValid][None]
    rgb_i = rgb_i[indValid][None]
    rgb_j = rgb_j[indValid][None]
    return depth, rgb_i, rgb_j

def backprojection(x, Z, Kinv): 
    # input image points need to be 3xN matrix
    # depth points need to be 1xN vector
    xy_h = Kinv @ x # xy homogeneous image coordinates
    xy = dehomogenize(xy_h)
    x, y = xy[0,:], xy[1,:]
    # multiply with depth to get XY homogeneous scene coordinates
    X = np.multiply(x,Z)
    Y = np.multiply(y,Z)
    # concatenate points in 3xN matrix
    X_opt = np.concatenate((X, Y, Z), axis=0)
    return homogenize(X_opt)


def init():
    T_oc = np.array([[0, -1,  0, 0],
                     [0,  0, -1, 0],
                     [1,  0,  0, 0],
                     [0,  0,  0, 1]])
    roll = 0
    pitch = 0.36
    yaw = 0.021
        
    R_z = np.array([[np.cos(yaw),  -np.sin(yaw), 0],
                     [np.sin(yaw), np.cos(yaw),  0],
                     [0,           0,            1]])
    
    R_y = np.array([[np.cos(pitch),  0, np.sin(pitch)],
                    [0,              1,             0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_x = np.array([[1, 0,                        0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll),   np.cos(roll)]])
    
    R = R_z @ R_y @ R_x
    t = np.array([0.18, 0.005, 0.36])[None]
    Rt = np.concatenate((R, t.T), axis=1)
    last = np.array([0, 0, 0, 1])[None]
    T_bc = np.vstack((Rt, last))
    
    T_bo = optical2body(T_oc, T_bc)
    
    K = np.array([[585.05108211, 0,            242.94140713],
                  [0,            585.05108211, 315.83800193],
                  [0,            0,                       1]])
    Kinv = np.linalg.inv(K)
    return Kinv, T_bo
    

#%%


TMAP = {}
TMAP['res']   = 0.05 #meters
TMAP['xmin']  = -35  #meters
TMAP['ymin']  = -35
TMAP['xmax']  =  35
TMAP['ymax']  =  35 
TMAP['sizex']  = int(np.ceil((TMAP['xmax'] - TMAP['xmin']) / TMAP['res'] + 1)) #cells
TMAP['sizey']  = int(np.ceil((TMAP['ymax'] - TMAP['ymin']) / TMAP['res'] + 1))
TMAP['map'] = np.zeros((TMAP['sizex'],TMAP['sizey'], 3),dtype=np.uint16) #DATA TYPE: char or int8
Kinv, T_bo = init()


#interval = 10
interval = 5
end = len(disp_stamps)

plt.ioff()
tic = time.time()

for ts in range(1, end, interval):
    img = Image.open(os.path.join(folder,filenames[ts]))
    
    # find closest rgb timestamp to disp timestamp     
    rgb_shifted_ts = rgb_stamps - disp_stamps[ts] 
    rgb_ind = np.argmin(np.abs(rgb_shifted_ts))
    rgb = Image.open(os.path.join(folder1,filenames1[rgb_ind]))
    
    # find closest encoder timestamp to disp timestamp
    enc_shifted_ts = encoder_stamps - disp_stamps[ts] 
    enc_ind = np.argmin(np.abs(enc_shifted_ts))
    
    
    # get depth from disp image and x,y pixel coordinates from rgb image
    disp = np.array(img.getdata(),  np.uint16).reshape(img.size[0], img.size[1])
    depth, rgb_i, rgb_j = get_RGBD(disp)

    # backproject points to optical frame, then convert to robot frame
    xy = np.vstack((rgb_i, rgb_j))
    xy = homogenize(xy)
    Xo = backprojection(xy, depth, Kinv) # transforms points to optical frame
    Xb = T_bo @ Xo # transforms points to robot body frame
    
    rgb_img = np.array(rgb.getdata(), np.uint16).reshape(rgb.size[0],rgb.size[1], 3)
    pix = rgb_img[rgb_i, rgb_j, :]
    
    # tranforms from robot body frame to world frame using, this uses PARTICLES['trajectory'] from SLAM 
    T_wb = body2world(trajectory[:,enc_ind])
    Xw = T_wb @ Xb
    Xw = dehomogenize(Xw)
    
    # thresholds invalid heights 
    z = Xw[2,:]
    indValid = np.logical_and((z < 0.45),(z > 0))
    Xw = Xw[:,indValid]
    pix = pix[:,indValid,:]

    TX_grid = np.ceil((Xw[0,:] - TMAP['xmin']) / TMAP['res']).astype(np.int16) - 1
    TY_grid = np.ceil((Xw[1,:] - TMAP['ymin']) / TMAP['res']).astype(np.int16) - 1

    TMAP['map'][TX_grid, TY_grid, :] = pix[0,:,:] 
    
toc = time.time()
print('time elapsed :', toc-tic)


fig = plt.figure()
plt.imshow(TMAP['map'])
plt.axis('off')
#plt.imsave('/Users/jdeguzman/Desktop/ECE276A_HW2/dataset20/texturemap/tmap20.png', TMAP['map'], dpi=300)
plt.show()