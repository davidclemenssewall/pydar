# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 09:32:44 2020

@author: d34763s
"""

import time
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\lidar_processing\\')
import pydar
import vtk

# %% init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_102519.RiSCAN'

project = pydar.Project(project_path, project_name)

project.apply_transforms(['sop'])

# %% Apply snowflake_filter_2

z_diff = 0.05
r_min = 5
N = 5

t0 = time.process_time()
project.apply_snowflake_filter_2(z_diff, N, r_min)
t1 = time.process_time()
print(t1-t0)

# %% Display

z_min = -3
z_max = -1.5

project.display_project(z_min, z_max)

# %% Write Scans

project.write_scans()


# %% Delete project

del project

# %% Load again this time read_scans

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_102519.RiSCAN'

project = pydar.Project(project_path, project_name)

project.read_scans()

project.apply_transforms(['sop'])

# %% Display

z_min = -3
z_max = -1.5

project.display_project(z_min, z_max)