#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 08:55:32 2021

@author: thayer
"""

import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% location

project_path = "/media/thayer/Data/mosaic_lidar/RS"
project_name = "mosaic_rs_170420.RiSCAN"

project = pydar.Project(project_path, project_name, 
                      import_mode='read_scan', las_fieldnames=['Points',
                        'PointId', 'Classification'], class_list='all')

project.read_transforms()
project.apply_transforms(['current_transform'])
project.load_labels()

# %% Display

z_min = -3
z_max = -1

project.display_project(z_min, z_max, show_labels=True)