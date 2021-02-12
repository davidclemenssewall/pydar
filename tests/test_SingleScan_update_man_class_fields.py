# -*- coding: utf-8 -*-
"""
test_SingleScan_update_man_class_fields.py

Created on Fri Jan 29 12:27:42 2021

@author: thayer
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar

# %% init

project_path = 'D:\\mosaic_lidar\\Snow2\\'
project_name = 'mosaic_02_110619.RiSCAN'
scan_name = 'ScanPos001'

ss = pydar.SingleScan(project_path, project_name, scan_name, read_scan=True)

ss.load_man_class()

# %% This is a bit of a contrived example but we'll change the dist field

# Look at it before changing
f, axs = plt.subplots(1, 2, figsize=(15, 8))

axs[0].scatter(x='Density', y='dist', data=ss.man_class)

# change all dist values to 100
ss.dsa_raw.PointData['dist'][:] = 100

# Update fields
ss.update_man_class_fields(update_fields=['dist'], update_trans=False)

# Plot again
axs[1].scatter(x='Density', y='dist', data=ss.man_class)

# %% And reverse, mostly just so we haven't stored bogus data

# Look at it before changing
f, axs = plt.subplots(1, 2, figsize=(15, 8))

axs[0].scatter(x='Density', y='dist', data=ss.man_class)

# change all dist values to 100
ss.add_dist()

# Update fields
ss.update_man_class_fields(update_fields=['dist'], update_trans=False)

# Plot again
axs[1].scatter(x='Density', y='dist', data=ss.man_class)

