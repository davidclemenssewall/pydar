#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_SingleScan_write_read_frozen_scan.py

Created on Mon Feb 21 10:49:47 2022

@author: thayer
"""

import json
import os
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

print(original_tree)

# %% Try writing the frozen scan

suffix = '_archive'

ss.write_scan(suffix=suffix, freeze=True)

# %% Check that we cannot write to that directory

print(os.access(os.path.join(project_path, project_name, 'npyfiles'+suffix,
                             scan_name), os.W_OK))

# %% Now delete ss and load from the written scan

#del ss

#ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
#                      'read_scan')

# %% Check that we loaded everything back in correctly

#original_tree==json.dumps(ss.raw_history_dict, indent=4)