#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_Project_import_mode.py

test that SingleScan import mode functionality is working correctly in Project

Created on Thu Mar  4 17:25:46 2021

@author: thayer
"""

import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% location

project_path = "/media/thayer/Data/mosaic_lidar/ROV"
project_name = "mosaic_rov_040120.RiSCAN"

# %% Old fashioned init

project = pydar.Project(project_path, project_name)

# produces a warning as desired

# %% New version

project = pydar.Project(project_path, project_name, import_mode=
                      'read_scan')

# that seems to work
# Before only creating cells when needed this goes 2.8 -> 25.7 GB
# Creating cells only as needed drops this to be 2.9 -> 23.7 GB (2GB)

# %% How about empty init

project = pydar.Project(project_path, project_name, import_mode=
                      'empty')

# %% What about all points from numpy as fast as we can

project = pydar.Project(project_path, project_name, import_mode=
                      'import_npy', create_id=False, las_fieldnames=['Points']
                      , class_list='all')