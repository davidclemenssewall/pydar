# -*- coding: utf-8 -*-
"""
Script to test Project class in pydar module

Created on Fri Sep 11 11:59:03 2020

@author: d34763s
"""

import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% Import project

project_path = '/media/thayer/Data/mosaic_lidar/Snow1/'

project_name = 'mosaic_01_101819.RiSCAN'

project = pydar.Project(project_path, project_name, import_mode='import_las')
print(project.project_date)

# %% Apply SOP transform to all SingleScans

project.apply_transforms(['sop'])

# %% Display

z_min = -3.5
z_max = -1.5

project.display_project(z_min, z_max)

# %% Display with scanners
z_min = 0
z_max = 2

project.display_project(z_min, z_max, show_scanners=True)
