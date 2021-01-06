# -*- coding: utf-8 -*-
"""
test_SingleScan_create_dimensionality_pdal.py

Test using pdal to create dimensions verticality, linarity, planarity, 
scattering.

Created on Tue Jan  5 14:07:00 2021

@author: d34763s
"""

import numpy as np
import vtk
import os
os.chdir('C:\\Users\\d34763s\\Desktop\\DavidCS\\PhD\\code\\pydar\\')
import pydar

# %% Test init

project_path = 'D:\\mosaic_lidar\\Snow1\\'
project_name = 'mosaic_01_040120.RiSCAN'
scan_name = 'ScanPos006'

ss = pydar.SingleScan(project_path, project_name, scan_name, read_scan=True)

# %% Add sop

ss.add_sop()
ss.apply_transforms(['sop'])

# %% Test function

ss.create_dimensionality_pdal()