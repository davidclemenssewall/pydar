#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_SingleScan_get_local_max.py

Test getting locally maximal values

Created on Thu Mar 10 05:02:39 2022

@author: thayer
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# Let's load a SingleScan
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_250120.RiSCAN' #'mosaic_rov_110320.RiSCAN'
scan_name = 'ScanPos001'

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
                      'read_scan', las_fieldnames=['Points', 'PointId',
                                                   'Classification'])

# %% Transform

ss.read_transform()
ss.apply_transforms(['current_transform'])

# %% Get Local max

z_threshold = -1.9
rmax = 0.5

arr_pts, = ss.get_local_max(z_threshold, rmax)

# %% Try getting with distance and z-sigma

local_max = ss.get_local_max(z_threshold, rmax, return_dist=True, return_zs
                             =True)

# %% Check that all local max are separated by rmax

from scipy.spatial import KDTree

tree = KDTree(local_max[0][:,:2])
pairs = tree.query_pairs(rmax, output_type='ndarray')

print(pairs)