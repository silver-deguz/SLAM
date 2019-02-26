#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:11:02 2019

@author: jdeguzman
"""
from map_utils import bresenham2D, mapCorrelation
from slam_functions import *
import numpy as np


#%%

# ---------------------- #
#       MAPPING          #
# ---------------------- #
   
def mapping(MAP, PARTICLES, z, T_bl):
    mu = PARTICLES['best'] 
    
    # returns lidar scans in homogeneous cartesian coords
    # this also filters out the noisy laser scans outside of [0.2, 30] meters
    Z0 = get_lidar_cartesiancoords(z) # 4xN matrix
    
    # transform lidar readings from lidar -> body, and then body -> world 
    T_wb = body2world(mu) 
    T_wl = T_wb @ T_bl    
    Z1 = T_wl @ Z0 # 4xN matrix, lidar scans in world coords

    # extract only x and y components of world coordinates
    X, Y = Z1[0,:], Z1[1,:]
    
    # convert robot position in world frame (in meters) to grid cells
    # this is the start point for raytracing
    xs_grid = np.ceil((mu[0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    ys_grid = np.ceil((mu[1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    # these are the end points for each ray
    xe_grid = np.ceil((X - MAP['xmin'])/ MAP['res']).astype(np.int16) - 1
    ye_grid = np.ceil((Y - MAP['ymin'])/ MAP['res']).astype(np.int16) - 1

    # raytracing to get occupied/free cells to update log-odds map
    # increment the observed occupied cells and decrement the free cells
    for i in range(len(X)):        
        ray = bresenham2D(xs_grid, ys_grid, xe_grid[i], ye_grid[i]).astype(int)
        MAP['map'][ray[0,:-1], ray[1,:-1]] += MAP['free'] 
        MAP['map'][ray[0,-1], ray[1,-1]] += MAP['occupied']
    
    # threshold log-odds values 
    MAP['map'] = np.clip(MAP['map'], -MAP['map_threshold'], MAP['map_threshold']) 
    
    
# ----------------------------------- #
#     LOCALIZATION - PREDICTION       #
# ----------------------------------- #
    
def motion_model(s_t0, tv, w, tau): 
    yaw = w * tau
    theta = s_t0[2,:] + yaw 
    x = s_t0[0,:] + ( tv * np.sinc(yaw/2) * np.cos(s_t0[2,:] + (yaw/2)) ) 
    y = s_t0[1,:] + ( tv * np.sinc(yaw/2) * np.sin(s_t0[2,:] + (yaw/2)) ) 
    
    # update state of particles
    s_t = np.vstack((x,y,theta)) # 3xN
    return s_t

def prediction(PARTICLES, tau, odom, w, noise=True):       
    # build motion model
    tv = get_displacement(odom, tau)
    mu = motion_model(PARTICLES['poses'], tv, w, tau) # size 3xN   

    # update pose, and store trajectory
    PARTICLES['poses'] = mu
    
    if noise:
        additive_noise = np.random.randn(2, PARTICLES['nums']) * 0.008
        additive_noise_theta = np.random.randn(1, PARTICLES['nums']) * 0.001
#        PARTICLES['poses'][0,:] += additive_noise[0,:] # noise in x
#        PARTICLES['poses'][1,:] += additive_noise[1,:] # noise in y 
        PARTICLES['poses'][2,:] += additive_noise_theta[0,:] # noise in theta 


# ----------------------------------- #
#        LOCALIZATION - UPDATE        #
# ----------------------------------- #
        
def get_mapcorrelation(MAP, Z_world):
     x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) # x-positions of each pixel of the map
     y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) # y-positions of each pixel of the map
     x_range = np.arange(-0.2,0.2+0.05,0.05)
     y_range = np.arange(-0.2,0.2+0.05,0.05)
     c = mapCorrelation(MAP['bmap'], x_im, y_im, Z_world, x_range, y_range) # use binary maps
     val, ind = np.max(c), np.argmax(c)
     return val, ind

def update(PARTICLES, MAP, z, T_bl):
    Z0 = get_lidar_cartesiancoords(z)
    
    # convert log-odds map to binary map
    MAP['bmap'] = 1 - (1 / (1+np.exp(MAP['map'])))
    MAP['bmap'][MAP['bmap'] >= MAP['bmap_threshold']] = 1
    MAP['bmap'][MAP['bmap'] < 0.3] = 0
    
    corr = np.zeros(PARTICLES['nums']) # store correlation values for all the particles
    
    for k in range(PARTICLES['nums']):
        mu = PARTICLES['poses'][:,k]
        T_wb = body2world(mu)
        
        # transform lidar scans from lidar -> world
        T_wl = T_wb @ T_bl    
        Z1 = T_wl @ Z0 
        
        X, Y = Z1[0,:], Z1[1,:] 
        Z_world = np.vstack((X,Y)) # 2xN array, lidar points in world coordinates
        max_val, max_ind = get_mapcorrelation(MAP, Z_world)
        corr[k] = max_val
    
    # update particle weights
    h = corr - np.max(corr) # softmax trick
    alphas = PARTICLES['weights'] * np.exp(h)
#    PARTICLES['weights'] = alphas / np.linalg.norm(alphas)
    PARTICLES['weights'] = alphas / np.sum(alphas)
        
    

     

