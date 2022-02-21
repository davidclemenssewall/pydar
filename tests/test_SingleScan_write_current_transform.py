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
import os
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

trans_suffix = '_temp'

# %% Check that we created a filter tree
original_tree = json.dumps(ss.transformed_history_dict["input_1"], indent=4)

# %% Try just writing the transform

ss.write_current_transform(suffix=trans_suffix)

# %% Now read the transform

ss.read_transform(suffix=trans_suffix)

# %% check that they match

print("History match?")
print(original_tree==json.dumps(ss.trans_history_dict['current_transform'],
                                indent=4))

print("Transform Differences?")
for i in range(4):
    for j in range(4):
        print(ss.transform.GetMatrix().GetElement(i, j) - ss.transform_dict[
            'current_transform'].GetMatrix().GetElement(i, j))

# %% Now try writing a frozen version

name = 'frozen'
ss.write_current_transform(name=name, suffix=trans_suffix, freeze=True)

# %% read the frozen version and examine history dict

ss.read_transform(name=name, suffix=trans_suffix)

print(json.dumps(ss.trans_history_dict[name],
                                indent=4))

# %% try overwriting

ss.write_current_transform(name=name, suffix=trans_suffix, freeze=False)
# that correctly raises an error

# %% overwrite_frozen=True
ss.write_current_transform(name=name, suffix=trans_suffix, freeze=False,
                           overwrite_frozen=True)

ss.read_transform(name=name, suffix=trans_suffix)

print(json.dumps(ss.trans_history_dict[name],
                                indent=4))

# %% Clean up directories

dirname = os.path.join(project_path, project_name, 'transforms' + trans_suffix, 
                       scan_name)

for f in os.listdir(dirname):
    os.remove(os.path.join(dirname, f))
os.rmdir(dirname)
os.rmdir(os.path.join(project_path, project_name, 'transforms' + trans_suffix))
