#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_SingleScan_write_current_transform.py

Test functionality for writing and reading transforms

Created on Thu Mar 18 14:14:24 2021

@author: thayer
"""

import numpy as np
import json
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

# %% Check that we created a filter tree
original_tree = json.dumps(ss.transformed_history_dict["input_1"], indent=4)

# %% Try just writing the scan

ss.write_current_transform()

# %% Now read the transform

ss.read_transform()

# %% check that they match

print("History match?")
print(original_tree==json.dumps(ss.trans_history_dict['current_transform'],
                                indent=4))

print("Transform Differences?")
for i in range(4):
    for j in range(4):
        print(ss.transform.GetMatrix().GetElement(i, j) - ss.transform_dict[
            'current_transform'].GetMatrix().GetElement(i, j))
