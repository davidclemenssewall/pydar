#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_Project_point_to_grid_average_image.py

Created on Wed Jun 23 16:55:45 2021

@author: thayer
"""

import json
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# Parameters for reconstruction
dx = 0.1
dy = 0.1

# %% load February 4 scan where we've defined the boundaries

project_path = "/media/thayer/Data/mosaic_lidar/ROV"
project_name = "mosaic_rov_040220.RiSCAN"


project = pydar.Project(project_path, project_name, import_mode=
                      'read_scan', las_fieldnames=['Points', 'PointId'],
                      class_list='all')

project.read_transforms()
project.apply_transforms(['current_transform'])

f = open('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/' +
         'lidar_research/rov_processing/leg_2_border.txt', 'r')
areapoints = json.load(f)#[project_name]
f.close()

cornercoords = project.areapoints_to_cornercoords(areapoints)

x0 = cornercoords[:,0].min()
y0 = cornercoords[:,1].min()
nx = int((cornercoords[:,0].max()-x0)/dx)
ny = int((cornercoords[:,1].max()-y0)/dy)
yaw = 0

del project

# %% load Jan. 19 scan

project_name = "mosaic_rov_190120.RiSCAN"
project = pydar.Project(project_path, project_name, import_mode=
                  'read_scan', las_fieldnames=['Points', 'PointId',
                    'Classification'],
                  class_list=[0, 2])

project.read_transforms()
project.apply_transforms(['current_transform'])

# %% convert to image

project.point_to_grid_average_image(nx, ny, dx, dy, x0, y0, yaw=yaw)

# %% examine

z_min = -3
z_max = -1

project.display_image(z_min, z_max)

# %% See if we can warp scalars with this many nan's...

project.display_image(z_min, z_max, warp_scalars=True)