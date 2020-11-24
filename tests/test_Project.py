# -*- coding: utf-8 -*-
"""
Script to test Project class in pydar module

Created on Fri Sep 11 11:59:03 2020

@author: d34763s
"""

import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar

# %% Test init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01b_061219.RiSCAN.RiSCAN.RiSCAN'

project = pydar.Project(project_path, project_name)
print(project.project_date)

# %% Add z offset and apply transforms

project.add_z_offset(4.5)
project.apply_transforms(['sop', 'z_offset'])

# %% Display

z_min = 0
z_max = 2

project.display_project(z_min, z_max)
