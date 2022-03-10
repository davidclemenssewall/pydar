#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_Project_create_local_max.py

Created on Thu Mar 10 10:21:15 2022

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

# Let's load a project
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_040120.RiSCAN'

project = pydar.Project(project_path, project_name, import_mode=
                      'read_scan', las_fieldnames=['Points', 'PointId',
                                                   'Classification'])

project.read_transforms()
project.apply_transforms(['current_transform'])

# %% Get local maxima

z_threshold = -1.9
rmax = 0.5

project.create_local_max(z_threshold, rmax)

# %% Try displaying

zmin = -3
zmax = -1

project.display_project(zmin, zmax, pdata_list=[['local_max']])

# %% Load image to try displaying on image

project.read_image(name='ft_ridge')

# %% Display on top of image

project.display_image(zmin, zmax, pdata_list=[['local_max']])

# %% Display on top of image, warp scalars

project.display_image(zmin, zmax, pdata_list=[['local_max']],
                      warp_scalars=True)