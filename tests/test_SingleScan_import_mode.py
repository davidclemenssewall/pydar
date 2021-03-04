#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_SingleScan_import_mode.py

test new import mode functionality in SingleScan init

Created on Thu Mar  4 10:47:49 2021

@author: thayer
"""

import sys
sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# %% location

project_path = "/media/thayer/Data/mosaic_lidar/ROV"
project_name = "mosaic_rov_190120.RiSCAN"
scan_name = "ScanPos004"

# %% Old fashioned init

ss = pydar.SingleScan(project_path, project_name, scan_name)

# produces a warning as desired

# %% New version

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
                      'read_scan')

# that seems to work

# %% How about empty init

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
                      'empty')

