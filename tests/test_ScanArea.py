# -*- coding: utf-8 -*-
"""
Test out registering a project and writing out an image to a file

Created on Mon Sep 14 10:20:17 2020

@author: d34763s
"""

import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar

# %% Test init

snow1 = pydar.ScanArea('D:\\mosaic_lidar\\Snow1\\')

# %% Add projects

snow1.add_project('mosaic_01_102019.RiSCAN')
snow1.add_project('mosaic_01_102519.RiSCAN')

# %% Compare reflectors between projects
snow1.compare_reflectors('mosaic_01_102019.RiSCAN', 'mosaic_01_102519.RiSCAN',
                         delaunay=True)

# %% Register Scan

# Looking at that plot I will pick a set of reflectors to use
reflector_list = ['r01', 'r02', 'r03', 'r09', 'r08']

snow1.register_project('mosaic_01_102019.RiSCAN', 'mosaic_01_102519.RiSCAN',
                       reflector_list)

# %% Examine

print(snow1.project_dict['mosaic_01_102519.RiSCAN'].tiepointlist.transforms)

z_min = -3
z_max = -1

snow1.project_dict['mosaic_01_102019.RiSCAN'].apply_transforms(['sop'])
snow1.project_dict['mosaic_01_102019.RiSCAN'].display_project(z_min, z_max)

# %% Now other project

snow1.project_dict['mosaic_01_102519.RiSCAN'].display_project(z_min, z_max)

# %% Now write image to file

z_min = -2.85
z_max = -1.95
focal_point = [150, 50, -50]
camera_position = [150, 50, 25]
image_scale = 500
upper_threshold = -1.3

snow1.project_dict['mosaic_01_102519.RiSCAN'].project_to_image(z_min, z_max,
                                                               focal_point,
                                                               camera_position,
                                                               image_scale,
                                                               upper_threshold=
                                                               upper_threshold)