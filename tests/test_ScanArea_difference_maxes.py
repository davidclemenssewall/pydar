#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ScanArea_difference_maxes.py

Created on Wed Mar 16 15:37:15 2022

@author: thayer
"""

import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# Let's load a project
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_names = ['mosaic_rov_040120.RiSCAN',
                 'mosaic_rov_250120.RiSCAN',
                 ]

scan_area = pydar.ScanArea(project_path, project_names, import_mode=
                      'read_scan', las_fieldnames=['Points', 'PointId',
                                                   'Classification'])

z_threshold = -1.9
rmax = 0.5

for project_name in project_names:
    project = scan_area.project_dict[project_name]
    project.read_transforms()
    project.apply_transforms(['current_transform'])
    project.create_local_max(z_threshold, rmax)
    
r_pair = 0.05

scan_area.difference_maxes(project_names[0], project_names[1], r_pair)

# %% Examine difference pdata

print(scan_area.max_difference_dict[(project_names[0], project_names[1])])