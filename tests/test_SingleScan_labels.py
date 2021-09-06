#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_SingleScan_labels.py

Test creating and adding to the labels dataframe under SingleScan.

Created on Wed Aug 25 15:11:06 2021

@author: thayer
"""

import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% location

project_path = "/media/thayer/Data/mosaic_lidar/Snow1"
project_name = "mosaic_01_111519.RiSCAN"
scan_name = 'ScanPos002'

ss = pydar.SingleScan(project_path, project_name, scan_name, 
                      import_mode='read_scan', las_fieldnames=['Points',
                        'PointId', 'Classification'], class_list='all')

ss.add_sop()
ss.apply_transforms(['sop'])

# %% explore labels

ss.load_labels()
print(ss.labels)

pos = ss.transform.GetPosition()

ss.add_label('test', 'temp', 5, pos[0], pos[1], pos[2])
ss.add_label('test', 'temp', 4, 1, 2, 3, transform='raw')

# %% Try getting all labels


df = ss.get_labels()