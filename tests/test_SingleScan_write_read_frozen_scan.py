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

suffix = '_temp'

ss.write_scan(suffix=suffix, freeze=True)


# %% Now delete ss and load from the written scan

del ss

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
                      'read_scan', suffix=suffix)

# %% Check that we loaded everything back in correctly

print(json.dumps(ss.raw_history_dict, indent=4))

# %% Try overwriting

ss.write_scan(suffix=suffix, freeze=True)

# that correctly raises an error that we are trying to overwrite a frozen scan

# %% overwrite_frozen

ss.write_scan(suffix=suffix, overwrite_frozen=True)

# %% Delete directory

dirname = os.path.join(project_path, project_name, 'npyfiles' + suffix, 
                       scan_name)

for f in os.listdir(dirname):
    os.remove(os.path.join(dirname, f))
os.rmdir(dirname)
os.rmdir(os.path.join(project_path, project_name, 'npyfiles' + suffix))
