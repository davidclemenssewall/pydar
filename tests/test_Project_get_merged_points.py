#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_Project_get_merged_points.py

Test the get_merged_points method

Created on Mon Mar 15 19:05:09 2021

@author: thayer
"""

import json
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% location

project_path = "/media/thayer/Data/mosaic_lidar/ROV"
project_name = "mosaic_rov_190120.RiSCAN"

# %% Load project

project = pydar.Project(project_path, project_name, import_mode='poly')

project.apply_transforms(['sop'])

# %% Try getting merged_points

pdata, history_dict = project.get_merged_points(history_dict=True)

# %% examine

print(json.dumps(history_dict, indent=4))

# %% Now see what happens if we create a linked version

pdata, history_dict = project.get_merged_points(port=True, history_dict=True)

# %% Try Changing one of the history dicts

project.scan_dict['ScanPos001'].filt_history_dict = 'Testing!!!'

print(json.dumps(history_dict, indent=4))