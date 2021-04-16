#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_SingleScan_apply_snowflake_filter_3.py

test functionality for our snowflake filter 3

Created on Tue Mar 30 12:05:59 2021

@author: thayer
"""

import time
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import sys
import platform
if platform.system()=='Windows':
    sys.path.append('C:/Users/d34763s/Desktop/DavidCS/PhD/code/pydar/')
else:
    sys.path.append('/home/thayer/Desktop/DavidCS/ubuntu_partition/code/pydar/')
import pydar

# Load a Single Scan
project_path = '/media/thayer/Data/mosaic_lidar/ROV/'
project_name = 'mosaic_rov_02_090520.RiSCAN'
scan_name = 'ScanPos005'

ss = pydar.SingleScan(project_path, project_name, scan_name, import_mode=
                      'read_scan', las_fieldnames=['Points', 'Classification',
                                                   'PointId'])

ss.add_sop()
ss.apply_transforms(['sop'])

# %% Display classification

ss.create_filter_pipeline()

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.AddActor(ss.actor)

renderWindow.AddRenderer(renderer)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

iren.Initialize()
renderWindow.Render()
iren.Start()

# %% Apply new filter
leafsize = 100
z_std_mult = 3.5

t0 = time.perf_counter()
ss.apply_snowflake_filter_3(z_std_mult, leafsize)
t1 = time.perf_counter()

print(t1-t0)

# %% Display classification

ss.create_filter_pipeline()

renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderer.AddActor(ss.actor)

renderWindow.AddRenderer(renderer)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renderWindow)

iren.Initialize()
renderWindow.Render()
iren.Start()
