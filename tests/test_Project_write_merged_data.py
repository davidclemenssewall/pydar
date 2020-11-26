# -*- coding: utf-8 -*-
"""
Test out writing merged polydata to a file

Created on Mon Sep 14 10:20:17 2020

@author: d34763s
"""

import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar

# %% Test init

snow1 = ScanArea('D:\\mosaic_lidar\\Snow1\\')

# %% Add z offset and apply transforms

project.add_z_offset(4)
project.apply_transforms(['sop', 'z_offset'])

# %% Write data

project.write_merged_points()