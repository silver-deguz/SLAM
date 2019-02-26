#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 01:10:25 2019

@author: jdeguzman
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random

def body2world(pose):
    theta = pose[2]
    x, y = pose[0], pose[1]
    P = np.array([[np.cos(theta), -np.sin(theta), 0, x],
                  [np.sin(theta),  np.cos(theta), 0, y],
                  [0,              0,             1, 0],
                  [0,              0,             0 ,1]])    
    return P

        
def get_lidar_cartesiancoords(z):
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0 
    
    # filter out noisy laser scan ranges outside of [0.1, 30] meters
    indValid = np.logical_and((z < 30),(z > 0.2))
    z = z[indValid]
    angles = angles[indValid]
    
    # xy position in the sensor frame
    xs_0 = z*np.cos(angles)[None]
    ys_0 = z*np.sin(angles)[None]
    zs_0 = np.zeros(xs_0.shape[1])[None]
    homog = np.ones(xs_0.shape[1])[None]
    
    # concatenate points in 4xN homogeneous matrix
    Z0 = np.concatenate((xs_0, ys_0, zs_0, homog), axis=0)
    return Z0
    

def get_displacement(odom, tau):
    counts = odom
#    counts = encoder_counts[:,ts] # 0.0022 = pi*d/360
    R_disp = ((counts[0] + counts[2]) / 2) * 0.0022 # average displacement of R wheels
    L_disp = ((counts[1] + counts[3]) / 2) * 0.0022 # average displacement of L weels
    tv = (R_disp + L_disp) / 2 
    return tv


def resampling(PARTICLES):
    N = PARTICLES['nums']
    
    # normalize weight and get cum sum
    weight_sum = np.cumsum(PARTICLES['weights'])
    weight_sum /= weight_sum[-1]
    
    # make N subdivisions, and chose a random position within each one
    rand = (np.linspace(0, N-1, N) + np.random.uniform(size=N)) / N
    new_sample = np.zeros(PARTICLES['poses'].shape)
    sample = 0
    ind = 0
    
    while(sample < N):
        while (weight_sum[ind] < rand[sample]):
            ind += 1
        new_sample[:,sample] = PARTICLES['poses'][:,ind]
        sample += 1
    PARTICLES['poses'] = new_sample
    PARTICLES['weights'] = np.ones(N) / N # renormalize
    

    
    