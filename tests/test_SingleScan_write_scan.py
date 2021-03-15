#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_SingleScan_write_scan.py

# Test our functionality for writing SingleScans to files

Created on Mon Mar 15 11:55:50 2021

@author: thayer
"""

import json
import vtk
import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% Test init

project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_190120.RiSCAN'
scan_name = 'ScanPos004'

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
                      'import_las')

# %% Check that we created a filter tree
original_tree = json.dumps(ss.raw_history_dict, indent=4)

# %% Try just writing the scan

ss.write_scan()

# %% Now delete ss and load from the written scan

del ss

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
                      'read_scan')

# %% Check that we loaded everything back in correctly

original_tree==json.dumps(ss.raw_history_dict, indent=4)