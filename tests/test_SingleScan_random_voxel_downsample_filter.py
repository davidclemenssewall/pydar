#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_SingleScan_random_voxel_downsample_filter.py

test random voxel downsampling

Created on Sun Mar 21 15:08:14 2021

@author: thayer
"""

import numpy as np
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% Test init

project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_190120.RiSCAN'
scan_name = 'ScanPos004'

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
                      'read_scan')

ss.add_sop()
ss.add_z_offset(3)

ss.apply_transforms(['sop', 'z_offset'])

# %% Test voxel downsampling

wx = 0.15
wy = 0.15
wz = 0.15

ss.random_voxel_downsample_filter(wx, wy, wz)
