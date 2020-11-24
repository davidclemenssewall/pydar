# -*- coding: utf-8 -*-
"""
test_Project_create_im_gaus.py

test create_im_gaus method.

Created on Mon Nov  2 14:49:49 2020

@author: d34763s
"""

import vtk
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar
from vtk.numpy_interface import dataset_adapter as dsa


# %% init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_102019.RiSCAN'


project = pydar.Project(project_path, project_name)

read_scans = True

if read_scans:
    project.read_scans()
    
else: 
    z_diff = 0.05
    r_min = 5
    N = 5
    project.apply_snowflake_filter_2(z_diff, N, r_min)
    project.write_scans()

project.apply_transforms(['sop'])

load_mesh = True

if load_mesh:
    project.read_mesh()
else:
    subgrid_x = 50
    subgrid_y = 50
    alpha = 5 # no gaps larger than 0.2
    overlap = 0.25
    
    project.merged_points_to_mesh(subgrid_x, subgrid_y, alpha=alpha,
                                  overlap=overlap)
    project.write_mesh()
    
# %% Display to check

z_min = -3
z_max = -1.5

project.display_project(z_min, z_max)

# %% Now create_empirical cdf

# Let's make these the same bounds as the image we'll compare with

nx = 1200
ny = 1200
dx = .1
dy = .1
x0 = 0 + 35
y0 = 133.5 + 45

project.mesh_to_image(nx, ny, dx, dy, x0, y0)

project.create_empirical_cdf_image(z_min, z_max)

# %% Now test create_im_gaus

project.create_im_gaus()

# %% and Display

project.image.GetPointData().SetActiveScalars('im_gaus')
project.display_image(-4, 4)