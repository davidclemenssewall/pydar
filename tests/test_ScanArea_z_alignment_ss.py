#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ScanArea_z_alignment_ss.py

Test functionality for aligning scans in successive projects by their gridded
z values.

Created on Mon Mar  8 16:09:57 2021

@author: thayer
"""

from collections import namedtuple
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar


# %% Registration list

Registration = namedtuple('Registration', ['project_name_0', 'project_name_1',
                                           'reflector_list', 'mode', 
                                           'yaw_angle'], 
                          defaults=[None, None, None])

project_names = ['mosaic_rov_190120.RiSCAN',
                 'mosaic_rov_250120.RiSCAN'
                 ]

registration_list = [Registration('mosaic_rov_250120.RiSCAN', 
                                  'mosaic_rov_250120.RiSCAN'),
                      Registration('mosaic_rov_250120.RiSCAN',
                                   'mosaic_rov_190120.RiSCAN',
                                   ['r28', 'r29', 'r30', 'r32', 'r34', 'r35',
                                    'r22'],
                                   'LS')
                      ]

# %% Init
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
suffix = 'slfsnow'


scan_area = pydar.ScanArea(project_path, project_names,
                       registration_list, import_mode='read_scan',
                       las_fieldnames=['Points', 'PointId', 'Classification'],
                       class_list='all', suffix=suffix)

scan_area.register_all()

# %% Test z_alignment_ss

frac_exceed_diff, diff, grid = scan_area.z_alignment_ss(project_names[1], 
                                             project_names[0], 'ScanPos001',
                                             w0=1, w1=1, min_pt_dens=25, 
                                             max_diff=0.05, 
                                             bin_reduc_op='mean', 
                                             return_grid=True)

# %% Examine
import matplotlib.pyplot as plt

f, axs = plt.subplots(1, 1)

axs.contourf(grid[:,0].reshape(diff.shape), 
             grid[:,1].reshape(diff.shape), grid[:,2].reshape(diff.shape))

axs.axis('equal')

# %%
import numpy as np

f, axs = plt.subplots(1, 1, figsize=(15, 15))

axs.contourf(grid[:,0].reshape(diff.shape), 
             grid[:,1].reshape(diff.shape), diff,
             antialiased=False, vmin=-0.02)

axs.axis('equal')

# %% 
f, axs = plt.subplots(1, 2, figsize=(15, 8))

axs[0].scatter(grid[:,0], diff.ravel(), alpha=0.05, marker='.')

axs[1].scatter(grid[:,1], diff.ravel(), alpha=0.05, marker='.')

# %% Fit a linear regression

diff = diff.ravel()
ind = np.logical_not(np.isnan(diff))

A = np.hstack((np.ones((ind.sum(),1)), grid[ind,:2]))
b = diff[ind, np.newaxis]

x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
